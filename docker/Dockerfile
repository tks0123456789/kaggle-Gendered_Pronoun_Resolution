FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime
RUN apt-get update && \
    apt-get install wget
RUN pip install spacy==2.1.3 && \
    pip install scikit-learn==0.20.3 && \
    pip install pandas==0.24.2 && \
    pip install argparse==1.4.0 && \
    pip install pytorch-pretrained-bert==0.6.1 && \
    python -c "from pytorch_pretrained_bert.modeling import BertModel;m=BertModel.from_pretrained('bert-large-cased');from pytorch_pretrained_bert import BertTokenizer;t = BertTokenizer.from_pretrained('bert-large-cased')" && \
    python -m spacy download en && \
    
    wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-development.tsv -q && \
    wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-test.tsv -q && \
    wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-validation.tsv -q