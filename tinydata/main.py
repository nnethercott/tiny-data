import argparse
import asyncio
import os

# https://medium.com/@yarusl42/asyncio-how-asyncio-works-part-2-33675c2c2f7d#:~:text=When%20you're%20writing%20production,focus%20on%20asynchronous%20I%2FO.
import uvloop

import tinydata.tinydata as td

from .clip import filter_directories

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

async def main(args):
    await td.run(args.topics, args.nsamples, args.dir)
    if args.filter:
        dirs = [os.path.join(args.dir, topic) for topic in args.topics]
        filter_directories(dirs = dirs, threshold = args.clip_similarity_threshold)

    # if args.filter -> filter

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--topics", help = "Space-delimited list of image classes", nargs='+', default=["dogs", "cats"])
    parser.add_argument("-n", "--nsamples",  help = "number of images to download per-class [default: 20]", type = int, default = 20)
    parser.add_argument("-d", "--dir", help = "name of directory to save to [default: images]", type = str, default = "images")
    parser.add_argument("-f", "--filter",  help ="filter images using CLIP cosine similarity", action='store_true')
    parser.add_argument("-c", "--clip_similarity_threshold", help="threshold for CLIP similarity used in optional filtering", type = float, default = 0.2)

    args = parser.parse_args()

    asyncio.run(main(args))
