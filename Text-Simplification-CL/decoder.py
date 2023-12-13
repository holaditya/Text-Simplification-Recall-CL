from run import *


def decode_live(base_path, best_model):
    # This loads  the model from the checkpoint and decodes the sentences from the user input
    print("Decoding sentences module executing...")
    logging.info(f"Decode module invoked.")
    _, _, _ = load_checkpt(base_path + best_model)
    print(f"Model loaded.")
    model.eval()

    print("Decoding Sentences...")
    should_continue = True
    while should_continue:
        print("Enter the sentence to be decoded: Enter 'exit' to exit")
        sent = input()
        if sent == "exit":
            should_continue = False
            break

        sent_tensor = tokenizer.encode_sent([sent])
        print(f"Sentence is : {sent}")
        print(f"Sentence tensor is: {sent_tensor}")
        predicted = model.generate(sent_tensor[0][0].to(device), attention_mask=sent_tensor[0][1].to(device),
                                   decoder_start_token_id=model.config.decoder.decoder_start_token_id)
        output = tokenizer.decode_sent_tokens([predicted.squeeze()])
        print("Simplified Sentence:", output)

    print("Okay, byee.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode in interactive environment")
    parser.add_argument("--base_path", type=str, default="./", help="Path to the checkpoint directory")
    parser.add_argument("--best_model", type=str, default="cp/int_adv/model_ckpt.pt",
                        help="Name of the best model checkpoint")

    args = parser.parse_args()
    base_path = args.base_path
    best_model = args.best_model
    decode_live(base_path, best_model)

