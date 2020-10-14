---
layout: post
title:  "Neural network from first principles with PyTorch"
date:   2020-09-29 21:09:46 +0100
categories: jekyll update
---

Thanks to fastai ([book](github.com/fastai/fastbook)\|[MOOC](fast.ai)\) for inspiring me to lay out this post. By making this content, I hope to cement my understanding and provide insight or clarification for any readers.

## Introduction

The most counterintuitve, brow-furrowing but stimulating thing about deep learning is its simplicity. We can describe it as 7 steps:

1. *Initialize* weights and biases - choose random values for model parameters.
2. *Predict* **Y** using **weights** * **X** + **bias**
3. Calculate *loss* - how good the predictions are 
4. Calculate *gradient* - the measure of how a unit change in the weight will change the value of loss
5. *Step* the weights according to this rate of change
6. Calculate metric to *validate* how well model is doing
7. *Repeat* steps 2 - 6
8. *Stop!*

Because of this simplicity, we can construct a neural network with little code - especially with the right toolkit. Let's jump in.

## What's so great about PyTorch?
On the surface, PyTorch looks a lot like Numpy with small naming changes - most notably, a numpy `array` is equivalent to a torch `tensor`. Under the hood, there is a key difference - PyTorch (and basically every other DL framework) uses CUDA, a programming platform that gives access to parallel computation on Nvidia GPUs. Numpy mostly uses C - static, compiled and faster than Python, but lacking ~*GPU MAGIC*~.
<br>
GPUs consist of hundreds of smaller cores - this makes them outstanding at doing lots of simple tasks in parallel. This highly parallelized computation is efficient for rendering graphics, but also maths - linear algebra, matrix operations and the like. This capacity for fast maths is what makes GPUs - and therefore CUDA and PyTorch - central to Deep Learning (see [here](https://www.infoworld.com/article/3299703/what-is-cuda-parallel-programming-for-gpus.html) for more info).

## 3s and 7s
To show it off, we will use a deep learning classic - the MNIST handwritten digit dataset. To make things simpler, we will frame this as a binary classification problem. We will train our model on 3s and 7s, and create an architecture that will return a probability for each outcome. OK - let's load it up!
```python
path = untar_data(URLs.MNIST_SAMPLE)

def get_list_of_tensors(path_ext):
	list_of_files = (path/path_ext).ls().sorted()
	list_of_tensors = [tensor(Image.open(o)) for o in files]
	return list_of_tensors

seven_tensors = get_list_of_tensors('train'/'7')
three_tensors = get_list_of_tensors('train'/'3')
seven_tensors_valid = get_list_of_tensors('valid'/'7')
three_tensors_valid = get_list_of_tensors('valid'/'3')

```
In order to run our parallelised PyTorch operations, we want to combine every image-tensor in the list into a single two-dimensional tensor. PyTorch has some handy functionalities for this:
> **_NOTE:_** In math-speak, such a 2-dimensional tensor is called a **rank 2 tensor**. I will be using this terminology from here

First, we `stack` our individual array tensors into one 3D tensor. as well as casting our tensors to `float` for upcoming math operations. Additionally, we will normalize pixel values to between 0 and 1 by dividing by 255. 
```python
def stack(list_of_tensors):
	stacked = torch.stack(list_of_tensors).float()/255
	return stacked
```
Then, we combine our 3 and 7 data into one single tensor with `cat`, and convert our rank 3 tensor into a more efficient rank 2 tensor using the reshape method `view`:
```python
def get_x(list_of_stacks):
	return torch.cat(list_of_stacks).view(-1,28*28)

train_x = get_x(stack(seven_tensors),stack(three_tensors))
valid_x = get_x(stack(seven_tensors_valid),stack(three_tensors_valid))
```
The parameters to `view()` can be confusing - the [docs](pytorch.org/docs/stable/tensor_view.html) say the arguments are `view(nrows,ncols)`.However, -1 is a special parameter that will find the appropriate number of rows to fit `ncol` columns of data - in our case, 1 row.

We also need to generate some independent variables for training:
```python
def get_y(positives,negatives):
	return tensor([1]*len(positives) + [0]*len(negatives)).unsqueeze(1)
train_y = get_y(three_tensors,seven_tensors)
valid_y = get_y(three_tensors_valid,seven_tensors_valid)
```

Next, we transform our tensors into a PyTorch `dataset` - a list of `(x,y)` tuples.
```python
dset=list(zip(train_x,train_y))
valid_dset = list(zip(valid_x,valid_y))
```

Finally, we will use a fastai DataLoader to batch this for us (remember how GPUS are good at parallelized computations):
```python
dl = DataLoader(dset,batch_size = 256)
valid_dl = DataLoader(valid_dset, batch_size = 256)
```
Now we have our data prepared, we can jump into building our neural network.


## 1. Initialize
We can use inbuilt PyTorch functions to easily initialize our architecture with very little code:
```python
simple_net = nn.Sequential(
	nn.Linear(28*28,30),
	nn.ReLU(),
	nn.Linear(30,1)
	)
```

## 2. Predict
To make predictions, we can simply call `simple_net` on our data i.e.:
```python
first_batch = first(dl)
simple_net(first_batch)
```

## 3. Loss
For a simple loss function, let's calculate how far away from truth (0 or 1) our predictions are, by normalizing these predictions between 0 and 1 with `sigmoid`. 
``` python
def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()
   ```
## 4. Gradient
Using our loss function, we can calculate the gradients using the `backward` method. This looks like magic, but really works because PyTorch is tracking the flow of tensors, allowing gradients to be calculated across this flow (see [docs](pytorch.org/docs/stable/autograd))
```python
def calc_grad(xb,yb,model):
	preds=model(xb)
	loss=mnist_loss(preds,yb)
	loss.backward()
```
Unfortunately, in practice,`loss.backward()` stores gradients, and will keep adding new gradients onto the existing value upon repeat calls of the function. Lets write some functionality to easily reset this for all parameters:
```py
class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None
opt = BasicOptim(simple_net.parameters(), 0.1)
 ```
 Also, see how nn.Sequential lets us call `parameters` on the whole architecture - this makes life a whole lot easier.
## 5. Step

Let's add a `step` method to our `BasicOptim` class:
```py
def step(self, *args, **kwargs):
    for p in self.params: p.data -= p.grad.data * self.lr
setattr(BasicOptim,'step',step)
```

## 6. Validate
To see how the model is performing, we want to see how our current model performs on the validation set. We simply calculate whether the sigmoid-transformed prediction is making the right 0/1 call, for every batch.

```py
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()


def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)
```
## 7. Repeat
Let's actually run these methods and repeat for each epoch:
```python
def train_epoch(model):
	for xb,yb in dl:
		calc_grad(xb,yb,model)
		opt.step()
		opt.zero_grad()


def train_model(model, epochs):
	for i in range(epochs):
		train_epoch(model)
		print(validate_epoch(model), end=' ')

train_model(linear_model,20)
```

## And now the fastai version...
Fastai saves boilerplate while still having a powerful and accessible low-level API. Having initialised our `simple_net` architecture, we can perform the equivalent of the 6 following steps with the following code:
```py
learn = Learner(dls, simple_net, opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)
learn.fit(20, 0.1)
```
Neat or what. That said, writing a deep learning algorithm in a lower-level framework such as PyTorch gives us a much more hands-on understanding of what's actually happening, which is key to being comfortable with the concepts.
