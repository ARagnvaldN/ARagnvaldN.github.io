---
layout: post
title: "Coding a neural network from scratch."
date: 2019-04-19
---

Contrary to popular belief the so-called Deep Learning is not actually that complicated. This is the post where I put all my thoughts about neural networks.  

I will go through the basic layers which have been around since the '80s one by one with a simple C++ implementation. The coding style might seem a bit odd since I will only use raw pointers, but I was curious to try my hand at some "dangerous" programming.  

All code is available [on my Github](http://github.com/aragnvaldn/nn_card/).

NB! This is a work in progress.  

## Inner product layer
Also known as "dot product" or "scalar multiplication".

```c
void inner_product(layer * input, layer * output)
{
  for (int outer = 0; outer < output->channels; ++outer) {

    float sum = 0;
    for (int inner = 0; inner < input->channels; ++inner) {
      for (int y = 0; y < input->shape; ++y) {
	 for (int x = 0; x < input->shape; ++x) {

	   sum += input->activations[inner * input->shape * input->shape
	 			     + y * input->shape
				     + x]
		  * output->weights[outer * input->channels 
					  * input->shape 
                                          * input->shape
				    + inner * input->shape * input->shape
				    + y * input->shape
				    + x];
	}
      }
    }
    // Apply bias and ReLU
    // Hinge on activation type and bias type
    float result = sum + output->bias[outer];
    if (output->relu) {
      if (result > 0.0f)
        output->activations[outer] = result;
    } else {
      output->activations[outer] = sum + output->bias[outer];
    }
  }
}

```
## Max pooling layer

```c
void max_pool(layer * input, layer * output)
{
  for (int c_out = 0; c_out < output->channels; ++c_out) {
    for (int h_out = 0; h_out < output->shape; ++h_out) {
      for (int w_out = 0; w_out < output->shape; ++w_out) {

      // Loop over inside of kernel
      float max = -10000.0f;
      for (int y_kernel = 0; y_kernel < output->kernel_size; ++y_kernel) {
        for (int x_kernel = 0; x_kernel < output->kernel_size; ++x_kernel) {
          float current_val = input->activations[c_out * input->shape
                                                       * input->shape
                                                 + (h_out * output->kernel_size
                                                    + y_kernel) * input->shape
                                                 + w_out * output->kernel_size
                                                 + x_kernel];
          if (current_val > max)
            max = current_val;
        }
      }
      output->activations[c_out * output->shape * output->shape
                          + h_out * output->shape
                          + w_out] = max;
      }
    }
  }
}
```

## Convolutional layer

```c
void convolution(layer * input, layer * output)
{
  int kernel_size = output->kernel_size;

  for (int c_outer = 0; c_outer < output->channels; ++c_outer) {
    for (int h_outer = 0; h_outer < output->shape; ++h_outer) {
      for (int w_outer = 0; w_outer < output->shape; ++w_outer) {
        
        float sum = 0.f;
        for (int c_inner = 0; c_inner < input->channels; ++c_inner) {
          for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y) {
            for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
    			    	    
              sum += input->activations[c_inner * input->shape * input->shape
                                        + (h_outer + kernel_y) * input->shape
                                        + w_outer + kernel_x]
                     * output->weights[c_outer * input->channels
                                               * kernel_size
                                               * kernel_size
                                       + c_inner * kernel_size * kernel_size
                                       + kernel_y * kernel_size
                                       + kernel_x];
            }
          }
        }
        output->activations[c_outer * output->shape * output->shape
                            + h_outer * output->shape
                            + w_outer] = sum + output->bias[c_outer];
      }
    }
  }
}
```

