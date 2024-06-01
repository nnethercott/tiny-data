import asyncio
import os
import time

import aiofiles
import aiohttp


# Function to read URLs from urls.txt
async def read_urls(file_path):
    async with aiofiles.open(file_path, mode='r') as file:
        urls = await file.readlines()
    return [url.strip() for url in urls][0:100]

# Function to download an image from a URL
async def download_image(session, url, download_path):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                filename = os.path.join(download_path, os.path.basename(url))
                async with aiofiles.open(filename, 'wb') as file:
                    content = await response.read()
                    await file.write(content)
                print(f"Downloaded: {url}")
            else:
                print(f"Failed to download {url} with status {response.status}")
    except Exception as e:
        print(f"Exception while downloading {url}: {str(e)}")

# Main function to orchestrate downloading
async def main(file_path, download_path):
    # Ensure download directory exists
    os.makedirs(download_path, exist_ok=True)

    # Read URLs
    urls = await read_urls(file_path)

    # Asynchronous HTTP requests session
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(download_image(session, url, download_path))
        
        # Await all download tasks
        await asyncio.gather(*tasks)

# Entry point
if __name__ == '__main__':
    file_path = 'urls.txt'  # File containing URLs
    download_path = 'downloads'  # Directory to save images

    start = time.time()
    asyncio.run(main(file_path, download_path))
    stop = time.time()
    print(stop-start)

