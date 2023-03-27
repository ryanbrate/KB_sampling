import json
import pathlib
import typing
from tqdm import tqdm


def main():

    # get samples excluding config.json
    samples_paths = filter(lambda path: path.name!="config.json", pathlib.Path(".").glob("*json"))

    # open the files
    for sample_path in tqdm(samples_paths):

        with open(sample_path, "r") as f:
            sample = json.load(f)

        try:
            # amend main body
            if isinstance(sample["text"]["p"], str):
                sample["text"]["p"] = correct_encoding(sample["text"]["p"])
            elif isinstance(sample["text"]["p"], list):
                l = []
                for s in sample["text"]["p"]:
                    l.append(correct_encoding(s))
                sample["text"]["p"] = l
            else:
                pass
        except:
            pass

        with open(sample_path, "w") as f:
            json.dump(sample, f, indent=4, ensure_ascii=False)

def correct_encoding(s:str):
    return s.encode("latin-1").decode("utf-8")



if __name__ == "__main__":
    main()
