from hasura_ndc import start
from hasura_ndc.function_connector import FunctionConnector
from pydantic import BaseModel
from typing import List, Optional
import os
import aiohttp
import asyncpg
import asyncio
import xml.etree.ElementTree as ET

connector = FunctionConnector()

class WikiEntry(BaseModel):
    entry_id: int
    content: str
    similarity: float

@connector.register_query
async def semanticSearchWiki(text: str, limit: Optional[int] = None, offset: Optional[int] = None) -> List[WikiEntry]:
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

        # Search for similar content
        searchQuery = """
            SELECT
                entry_id,
                content,
                1 - (embedding <=> $1::vector) as similarity
            FROM wiki_entries
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
        """

        if limit is not None:
            searchQuery += f" LIMIT {limit}"
            if offset is not None:
                searchQuery += f" OFFSET {offset}"
        else:
            searchQuery += " LIMIT 20"

        results = await conn.fetch(searchQuery, formattedEmbedding)

        # Map the results
        wikiEntries = [
            WikiEntry(
                entry_id=row['entry_id'],
                content=row['content'],
                similarity=row['similarity']
            ) for row in results
        ]

        await conn.close()
        return wikiEntries

    except Exception as e:
        print(f"Error performing semantic search: {e}")
        return []

@connector.register_mutation
async def vectorizeWikiContent() -> str:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pg_connection_uri = os.environ.get("PG_CONNECTION_URI")

    try:
        # Parse XML files
        wiki_entries = []
        for xml_file in ['Data/yakuza_pages_current.xml', 'Data/yakuza_pages_structure.xml']:  # Adjust filenames as needed
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for entry in root.findall('.//entry'):  # Adjust XML path based on your structure
                content = entry.find('content').text  # Adjust based on your XML structure
                wiki_entries.append(content)

        # Create database table if it doesn't exist
        conn = await asyncpg.connect(pg_connection_uri)
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS wiki_entries (
                entry_id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector(3072)  -- for text-embedding-3-large
            )
        ''')

        # Process entries in batches
        batchSize = 100
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        }

        for i in range(0, len(wiki_entries), batchSize):
            batch = wiki_entries[i:i+batchSize]

            async def process_entry(content):
                payload = {
                    "input": content,
                    "model": "text-embedding-3-large",
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload) as response:
                        embeddingData = await response.json()
                return {
                    'content': content,
                    'embedding': embeddingData['data'][0]['embedding']
                }

            tasks = [process_entry(content) for content in batch]
            processed_entries = await asyncio.gather(*tasks)

            # Insert entries and embeddings
            for entry in processed_entries:
                formattedEmbedding = '[' + ','.join(map(str, entry['embedding'])) + ']'
                await conn.execute(
                    'INSERT INTO wiki_entries (content, embedding) VALUES ($1, $2::vector)',
                    entry['content'],
                    formattedEmbedding
                )

            print(f"Processed {min(i + batchSize, len(wiki_entries))} out of {len(wiki_entries)} entries")

        await conn.close()
        return "SUCCESS"

    except Exception as e:
        print(f"Error vectorizing wiki content: {e}")
        raise e

if __name__ == "__main__":
    start(connector)