#FROM python:3.9.6
#FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

WORKDIR /

COPY . /

RUN pip install bert_score==0.3.9
RUN pip install datasets==1.9.0

CMD ["/bin/bash"]
