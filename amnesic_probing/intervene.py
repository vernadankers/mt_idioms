import sys
import random
import logging
import pickle
import sacrebleu
import numpy as np
import torch
from transformers import MarianTokenizer, MarianMTModel
sys.path.append('../data/')
from data import extract_sentences
from probing import set_seed
from classifier import Classifier

logging.getLogger().setLevel(logging.INFO)


def generate(model, tokenizer, sentence, projection=False, attention_projection_indices=None,
             attention_projection_matrices=[None]*7, hidden_projection_indices=None,
             hidden_projection_matrices=[None]*7, gather_attention=False):
    """
    Generate translations while projecting if projection is true.

    Args:
        - model
        - tokenizer
        - sentence (str)
        - projection (bool)
        - attention_projection_indices
        - attention_projection_matrices
        - hidden_projection_indices
        - hidden_projection_matrices
        - gather_attention (bool)

    Returns:
        - translation (str)
        - Attention matrix
    """
    batch = tokenizer.prepare_seq2seq_batch(src_texts=[sentence], return_tensors="pt")
    if torch.cuda.is_available():
        batch["input_ids"] = batch["input_ids"].cuda()
        batch["attention_mask"] = batch["attention_mask"].cuda()

    if torch.cuda.is_available():
        for i, matrix in enumerate(attention_projection_matrices):
            if matrix is not None:
                attention_projection_matrices[i] = matrix.cuda()
        for i, matrix in enumerate(hidden_projection_matrices):
            if matrix is not None:
                hidden_projection_matrices[i] = matrix.cuda()

    decoder_outputs = model.generate(
        **batch, beam_size=5, return_dict_in_generate=True,
        attention_projection_indices=attention_projection_indices,
        attention_projection_matrix=attention_projection_matrices,
        hidden_projection_matrix=hidden_projection_matrices,
        hidden_projection_indices=hidden_projection_indices,
        output_attentions=gather_attention)

    if gather_attention:
        attention = [att[0].cpu().detach().numpy()
                     for att in decoder_outputs["encoder_attentions"]]
    else:
        attention = None
    tgt_tokens = tokenizer.convert_ids_to_tokens(decoder_outputs["sequences"][0])
    return "".join(tgt_tokens[1:]).replace("▁", " ").strip(), attention


def main(data, attention_projection_matrices, hidden_projection_matrices,
         baseline=False, attention_labels=[1],
         hidden_labels=[1], gather_attention=False):
    """
    
    Args:
        - data
        - attention_projection_matrices
        - hidden_projection_matrices
        - baseline (bool)
        - attention_labels
        - hidden_labels
        - gather_attention

    Returns:

    """
    # Load the model and tokenizer
    model_version = 'Helsinki-NLP/opus-mt-en-nl'
    model = MarianMTModel.from_pretrained(model_version)
    tokenizer = MarianTokenizer.from_pretrained(model_version)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # Load the classifier
    classifier = Classifier("../data/idiom_keywords_translated.tsv")

    # Prune the data to only consider paraphrases
    all_without, all_with, attention = [], [], []
    idiom_scores = dict()

    maxi = max(list(data.keys()))
    for j in sorted(list(data.keys())):
        if not data[j]:
            continue
        idiom = data[j][0].idiom
        idiom_equal_labels = []

        for k, s in enumerate(data[j]):
            if (k + 1) % 20 == 0:
                logging.info(f"Sample {k} / {len(data[j])}")

            # Translate without the projection matrices
            without_projection, attention_without = generate(
                model, tokenizer, s.sentence, gather_attention=gather_attention)

            # Translate with the projection matrices
            with_projection, attention_with = generate(
                model, tokenizer, s.sentence, projection=True,
                attention_projection_indices=s.index_select(attention_labels, no_subtokens=False),
                attention_projection_matrices=attention_projection_matrices,
                hidden_projection_indices=s.index_select(hidden_labels, no_subtokens=False),
                hidden_projection_matrices=hidden_projection_matrices,
                gather_attention=gather_attention)

            # Collect the indices of PIE & context
            if gather_attention:
                noun_pie_index = s.index_select(1, tags=["NOUN"])
                noun_context_index = s.index_select(0, tags=["NOUN"], neighbours_only=True)
                all_pie_index = s.index_select(1, tags=[])
                all_context_index = s.index_select(0, tags=[], neighbours_only=True)

            old_label = classifier(s.idiom, without_projection)
            new_label = classifier(s.idiom, with_projection)
            if old_label == "paraphrase":
                idiom_equal_labels.append(new_label)

                # Store attention
                if gather_attention:
                    attention.append((
                        attention_without, attention_with, noun_pie_index,
                        noun_context_index, all_pie_index, all_context_index))
 
                all_with.append(with_projection)
                all_without.append(without_projection)
        if not idiom_equal_labels:
            continue

        # Score this idiom with BLEU and the percentage of change
        score = sacrebleu.corpus_bleu(all_with, [all_without]).score
        idiom_scores[idiom] = 1 - (idiom_equal_labels.count('paraphrase') / len(idiom_equal_labels))
        logging.info(f"Idiom {j}/{maxi}, BLEU={score:.3f}, %={idiom_scores[idiom]:.3f}"    )

    print(idiom_scores)
    bleu_score = sacrebleu.corpus_bleu(all_with, [all_without]).score
    return bleu_score, idiom_scores, attention


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[])
    parser.add_argument("--attention_layers", type=int, nargs='+', default=[])
    parser.add_argument("--hidden_labels", type=int, nargs='+', default=[1])
    parser.add_argument("--attention_labels", type=int, nargs="+", default=[1])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--baseline", type=bool, default=False)
    parser.add_argument("--filename", type=str, default="attention.pickle")
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--gather_attention", action="store_true")
    args = parser.parse_args()

    set_seed(1)
    # Load all hidden representations of idioms
    data = dict()
    for i in range(args.start, args.stop, args.step):
        if (i + 1) % 50 == 0:
            logging.info(f"Sample {i/args.step:.0f} / {(args.stop-args.start)/args.step:.0f}")
        samples = extract_sentences([i], use_tqdm=False)
        for s in samples:
            if s.translation_label == "paraphrase" and s.magpie_label == "figurative":
                s.label = 1
            else:
                s.label = None
        samples = [s for s in samples if s.label is not None]
        if samples:
            data[i] = samples

    indices = list(data.keys())
    random.shuffle(indices)

    n = int(len(indices)/5)
    fold_1 = indices[:n]
    fold_2 = indices[n:n*2]
    fold_3 = indices[n*2:n*3]
    fold_4 = indices[n*3:n*4]
    fold_5 = indices[n*4:]

    assert 0 not in args.attention_layers, "Cannot intervene om embs with attention queries!"

    bleus = []
    all_idioms = dict()
    for fold in args.folds:
        attention_projection_matrices = []
        hidden_projection_matrices = []

        if fold == 0:
            test = fold_1
        elif fold == 1:
            test = fold_2
        elif fold == 2:
            test = fold_3
        elif fold == 3:
            test = fold_4
        elif fold == 4:
            test = fold_5

        for layer in range(7):
            if layer in args.hidden_layers:
                P = pickle.load(
                    open(f"projection_matrices/hidden_fold={fold}_layer" + 
                         f"={layer}_baseline={args.baseline}.pickle", 'rb'))
            else:
                P = None
            hidden_projection_matrices.append(P)
            if layer in args.attention_layers:
                P = pickle.load(
                    open(f"projection_matrices/attention_fold={fold}_layer" +
                         f"={layer}_baseline={args.baseline}.pickle", 'rb'))
            else:
                P = None
            attention_projection_matrices.append(P)

        bleu, idiom_scores, attention = main(
            {i: data[i] for i in test}, attention_projection_matrices,
            hidden_projection_matrices, args.baseline, args.attention_labels,
            args.hidden_labels, args.gather_attention)

        for x in idiom_scores:
            all_idioms[x] = idiom_scores[x]

        bleus.append(bleu)
        if args.gather_attention:
            pickle.dump(attention, open(args.filename, 'wb'))

    percentage = np.mean(list(all_idioms.values()))
    logging.info(f"BLEU = {np.mean(bleus):.3f}, % idioms = {percentage}")
