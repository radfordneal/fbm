
# Commands to train a neural network on the binary response problem using
# gradent descent training.

net-spec blog.gd 2 15 1 / ih=+ bh=+ ho=+ bo=+
model-spec blog.gd binary

data-spec blog.gd 2 1 2 / bdata.train . 

net-gd blog.gd 100000 1000 / 0.4 batch

net-pred nmpa blog.gd 100000 / bdata.test .
