[Here I provide some modified examples taken from Torch7 demos. In the future I will attempt to make a PR to torch-demos repo. Visit the original 
git page for more demos and tutorials:  https://github.com/torch/demos and https://github.com/torch/tutorials]
<br />
1. Train CIFAR and Visualize CNN Weights and Responces <br />
Install the following packages: <br />
luarocks install nn <br />
luarocks install cunn <br /> (Optional, only if you want to run on the GPU)

<br />
Run the demo:
<br /> On CPU<br />
qlua cifar_training.lua
<br /> On GPU <br />
qlua cifar_training.lua -cuda
<br />
Or with visualization enabled:
<br />
qlua cifar_training.lua -visualize

