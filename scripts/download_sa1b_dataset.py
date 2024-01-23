from concurrent.futures import ThreadPoolExecutor
import requests
import tarfile
import os
import argparse


def download_and_extract_tar_file(url, path):
    response = requests.get(url, stream=True)
    with tarfile.open(fileobj=response.raw, mode="r|*") as tar:
        tar.extractall(path=path)


def download_sa1b_chunk_txt(directory, filename="data_urls.txt"):
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        txt_url = "https://scontent.fsgn5-6.fna.fbcdn.net/m1/v/t6/An8MNcSV8eixKBYJ2kyw6sfPh-J9U4tH2BV7uPzibNa0pu4uHi6fyXdlbADVO4nfvsWpTwR8B0usCARHTz33cBQNrC0kWZsD1MbBWjw.txt?ccb=10-5&oh=00_AfAh6HfOXvjS7AE8ocKiKk2uaYla9-hGn2TdD-oGIjhzzQ&oe=65D6B798&_nc_sid=0fdd51"
        response = requests.get(txt_url, stream=True)
        with open(path, mode="wb") as f:
            for chunk in response.raw.stream(1024, decode_content=False):
                if chunk:
                    f.write(chunk)
                    f.flush()

    url_dict = {}
    with open(path, mode="r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            file_name, cdn_link = line.strip().split("\t")
            chunk = file_name.split(".")[0]
            url_dict[chunk] = cdn_link

    return url_dict


def download_sa1b_dataset(out_dir, chunks):
    assert len(chunks) % 2 == 0
    selected_chunks = set()
    for i in range(0, len(chunks), 2):
        start = chunks[i]
        end = chunks[i + 1]
        for j in range(start, end):
            selected_chunks.add(j)

    url_dict = download_sa1b_chunk_txt(out_dir)
    cdn_links = []
    for i in selected_chunks:
        chunk_name = f"sa_{str(i).zfill(6)}"
        if chunk_name in url_dict:
            cdn_links.append(url_dict[chunk_name])

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with ThreadPoolExecutor() as exe:
        exe.map(download_and_extract_tar_file, cdn_links, [out_dir] * len(cdn_links))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", "-o", type=str, default="data/sa1b")
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        help="Chunk ids to download, should be in the format start1 end1 start2 end2 .... Then all chunks in range [start1, end1), [start2, end2), ... will be downloaded",
    )
    args = parser.parse_args()

    download_sa1b_dataset(args.out, args.chunks)
