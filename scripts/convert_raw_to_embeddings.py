from .utils.model_init import init_subject_model
import h5py
import torch
from conllu import parse_incr


def create_output_filename(input_path: str, output_dir: str, model_name: str) -> str:
    """
    Create the output filename for the embeddings
    Inputs:
    * input_path: str: path to the input file
    * output_dir: str: path to the output directory
    * model_name: str: name of the model
    Outputs:
    str: the output filename
    """
    basename = input_path.split("/")[-1].split(".")[0]
    model_name = model_name.replace("/", "_")
    return f"{output_dir}/{model_name}_{basename}.h5"


def convert_raw_to_embeddings(
    model_name: str,
    model_type: str,
    input_path: str,
    output_path: str,
    device: str = "cpu",
):

    model_init_dict = init_subject_model(
        model_name=model_name, model_type=model_type, device=device
    )

    text_model = model_init_dict["model_text"]
    tokenizer = model_init_dict["tokenizer"]
    text_config = model_init_dict["config_text"]

    num_layers = text_config.num_hidden_layers + 1  # +1 for the embeddings
    feature_count = text_config.hidden_size

    intput_path_split = input_path.split(".")
    input_text = []
    if intput_path_split[-1] == "conllu":
        with open(input_path, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                input_text.append(tokenlist)
    else:
        input_text = open(input_path).readlines()

    problematic_token_gang = 0
    total = len(input_text)
    with h5py.File(output_path, "w") as fout:
        for index, tokenlist in enumerate(input_text):
            line = tokenlist.metadata["text"]
            line = line.strip()
            # tokenize and truncate if length is greater than model max length
            tokenized = tokenizer(
                line,
                return_tensors="pt",
                truncation=True,
                max_length=text_config.max_position_embeddings,
            ).to(device=device)

            if tokenized.word_ids()[-2] + 1 != len(tokenlist):
                print(tokenized.word_ids(), len(tokenlist))
                print(tokenizer.tokenize(line), tokenlist)
                problematic_token_gang += 1
                for i in range(len(tokenlist)):
                    try:
                        local_list = []
                        for j, k in enumerate(tokenized.word_ids()):
                            if k == i:
                                local_list.append(tokenizer.tokenize(line)[j-1])
                        print(tokenlist[i], local_list, tokenlist[i]['upos'])
                    except:
                        print(tokenlist[i])

            with torch.no_grad():
                encoded_layers = text_model(
                    **tokenized, output_hidden_states=True
                ).hidden_states

            dset = fout.create_dataset(
                str(index), (num_layers, len(tokenized.input_ids[0]), feature_count)
            )

            dset[:, :, :] = torch.stack(encoded_layers).squeeze().cpu().numpy()
    print(problematic_token_gang, total, problematic_token_gang / total)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model to use for embeddings",
        required=True,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="The type of the model to use for embeddings",
        required=True,
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="The path to the input file to convert to embeddings",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="The device to use for embeddings",
        default="cpu",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="The path to the output file to save the embeddings",
        default=None,
    )

    args = parser.parse_args()
    output_path = create_output_filename(
        input_path=args.input_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )
    convert_raw_to_embeddings(
        model_name=args.model_name,
        model_type=args.model_type,
        input_path=args.input_path,
        output_path=output_path,
        device=args.device,
    )
