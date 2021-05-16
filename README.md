# mt_idioms

`transformers==4.3.0`, with `modeling_marian.py` replaced with the adapted version from this repository.

### Install
To clone `transformers` and replace MarianMT with our adapted version, carry out the following instructions:
```
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 800f385d7808262946987ce91c158186649ec954
cd ..
mv modeling_marian.py transformers/src/transformers/models/marian/modeling_marian.py
cd transformers
pip install -e .
```

### Experiments
- Section 4: See the `data` folder.
- Section 5: See the `attention` folder.
- Section 6.1: See the `probing` folder.
- Section 6.2 and 6.3: See the `hidden_representations` folder.
- Section 7: See the `wsd` folder.
- Appendix A: See the `attention` folder.
- Appendix B: See the `hidden_representations` folder.
- Appendix C: See the `wsd` folder.

This repo uses the [commit message template for humans](https://github.com/Kaleidophon/commit-template-for-humans).
