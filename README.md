# Real Time Recurrent Learnint(RTRL)

RTRL computes the recurrent parts of the gradients during the forward pass already. Thus, activations don't have to be stored along the whole sequence. Consequently, the complexity of RTRL is independent of the sequence length T. So it can be a good alternative for a very long sequences. 

At every time step t, RTRL must store the derivatives for computing the gradients for the next time step t + 1. This makes the complexity of RTRL independent of the sequence length T while using correct (as opposed to truncated BPTT) gradients at each time step.

 RTRL has a computational complexity of O(TI<sup>4</sup>). This is intractable for most RNNs of reasonable size.