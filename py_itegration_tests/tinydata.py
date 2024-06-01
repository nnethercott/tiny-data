import argparse
import asyncio

# https://medium.com/@yarusl42/asyncio-how-asyncio-works-part-2-33675c2c2f7d#:~:text=When%20you're%20writing%20production,focus%20on%20asynchronous%20I%2FO.
import uvloop

import tiny_data as td

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

async def main(args):
    await td.run(args.topics, args.nsamples, args.dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--topics", nargs='+', default=[])
    parser.add_argument("-n", "--nsamples",  type = int, default = 10)
    parser.add_argument("-d", "--dir",  type = str, default = "images")

    args = parser.parse_args()

    asyncio.run(main(args))
