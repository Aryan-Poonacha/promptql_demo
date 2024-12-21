from hasura_ndc import start
from hasura_ndc.function_connector import FunctionConnector
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import aiohttp
import asyncpg
import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime

connector = FunctionConnector()

class WikiPageRow(BaseModel):
    page_id: int
    title: str
    namespace: int
    content: str
    similarity: Optional[float] = None
    last_modified: datetime

async def parse_mediawiki_xml(xml_path: str):
    """Parse the MediaWiki XML dump and return structured page data"""
    try:
        # Register the MediaWiki XML namespace
        ET.register_namespace('', "http://www.mediawiki.org/xml/export-0.11/")
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract site information
        siteinfo = root.find('{http://www.mediawiki.org/xml/export-0.11/}siteinfo')
        site_name = siteinfo.find('{http://www.mediawiki.org/xml/export-0.11/}sitename').text
        print(f"Processing wiki: {site_name}")
        
        pages = []
        for page in root.findall('{http://www.mediawiki.org/xml/export-0.11/}page'):
            try:
                page_id = int(page.find('{http://www.mediawiki.org/xml/export-0.11/}id').text)
                title = page.find('{http://www.mediawiki.org/xml/export-0.11/}title').text
                namespace = int(page.find('{http://www.mediawiki.org/xml/export-0.11/}ns').text)
                
                # Get the latest revision
                revision = page.find('{http://www.mediawiki.org/xml/export-0.11/}revision')
                if revision is not None:
                    content = revision.find('{http://www.mediawiki.org/xml/export-0.11/}text').text or ""
                    timestamp_str = revision.find('{http://www.mediawiki.org/xml/export-0.11/}timestamp').text
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    # Only include pages with actual content
                    if content and len(content.strip()) > 0:
                        pages.append({
                            'page_id': page_id,
                            'title': title,
                            'namespace': namespace,
                            'content': content,
                            'last_modified': timestamp
                        })
            except Exception as e:
                print(f"Error processing page {title}: {e}")
                continue
        
        return pages
    except Exception as e:
        print(f"Error parsing XML: {e}")
        raise e

@connector.register_query
async def semanticSearchWiki(text: str, namespace: Optional[int] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> List[WikiPageRow]:
    """Search wiki pages using semantic similarity with optional namespace filtering"""
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pg_connection_uri = os.environ.get("PG_CONNECTION_URI")

    try:
        # Generate embedding for the search text
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        }
        payload = {
            "input": text,
            "model": "text-embedding-3-large",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload) as response:
                embeddingData = await response.json()

        embedding = embeddingData['data'][0]['embedding']
        formattedEmbedding = '[' + ','.join(map(str, embedding)) + ']'

        # Connect to the database
        conn = await asyncpg.connect(pg_connection_uri)

        # Base query with cosine similarity
        searchQuery = """
            SELECT
                page_id,
                title,
                namespace,
                content,
                last_modified,
                1 - (embedding <=> $1::vector) as similarity
            FROM WikiPages
            WHERE embedding IS NOT NULL
        """
        
        # Add namespace filter if specified
        if namespace is not None:
            searchQuery += f" AND namespace = {namespace}"
        
        # Add ordering and limits
        searchQuery += " ORDER BY embedding <=> $1::vector"
        
        if limit is not None:
            searchQuery += f" LIMIT {limit}"
            if offset is not None:
                searchQuery += f" OFFSET {offset}"
        else:
            searchQuery += " LIMIT 20"

        results = await conn.fetch(searchQuery, formattedEmbedding)

        # Map the results to the expected WikiPageRow interface
        wikiRows = [
            WikiPageRow(
                page_id=row['page_id'],
                title=row['title'],
                namespace=row['namespace'],
                content=row['content'],
                similarity=float(row['similarity']),
                last_modified=row['last_modified']
            ) for row in results
        ]

        await conn.close()
        return wikiRows

    except Exception as e:
        print(f"Error performing semantic search: {e}")
        return []

@connector.register_mutation
async def vectorizeWiki(xml_path: str) -> str:
    """Process MediaWiki XML dump and generate embeddings for all pages"""
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pg_connection_uri = os.environ.get("PG_CONNECTION_URI")

    try:
        # Parse XML file
        pages = await parse_mediawiki_xml(xml_path)
        
        # Connect to database
        conn = await asyncpg.connect(pg_connection_uri)

        # Create table if it doesn't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS WikiPages (
                page_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                namespace INTEGER NOT NULL,
                content TEXT,
                last_modified TIMESTAMP WITH TIME ZONE,
                embedding vector(3072)
            )
        """)

        # Process pages in batches
        batchSize = 50
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        }

        for i in range(0, len(pages), batchSize):
            batch = pages[i:i + batchSize]

            async def process_page(page):
                # Generate embedding for page content
                # Combine title and content for better semantic understanding
                text_to_embed = f"{page['title']}\n\n{page['content']}"
                
                payload = {
                    "input": text_to_embed,
                    "model": "text-embedding-3-large",
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload) as response:
                        embeddingData = await response.json()
                
                embedding = embeddingData['data'][0]['embedding']
                return {
                    **page,
                    'embedding': embedding
                }

            # Process batch concurrently
            tasks = [process_page(page) for page in batch]
            processed_pages = await asyncio.gather(*tasks)

            # Bulk insert/update pages
            for page in processed_pages:
                formattedEmbedding = '[' + ','.join(map(str, page['embedding'])) + ']'
                await conn.execute("""
                    INSERT INTO WikiPages 
                        (page_id, title, namespace, content, last_modified, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::vector)
                    ON CONFLICT (page_id) 
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        namespace = EXCLUDED.namespace,
                        content = EXCLUDED.content,
                        last_modified = EXCLUDED.last_modified,
                        embedding = EXCLUDED.embedding
                """, 
                    page['page_id'], 
                    page['title'],
                    page['namespace'],
                    page['content'],
                    page['last_modified'],
                    formattedEmbedding
                )

            print(f"Processed {min(i + batchSize, len(pages))} out of {len(pages)} pages")

        await conn.close()
        return "SUCCESS"

    except Exception as e:
        print(f"Error vectorizing wiki pages: {e}")
        raise e

if __name__ == "__main__":
    start(connector)