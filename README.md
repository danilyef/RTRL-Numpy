#Real Time Recurrent Learnint(RTRL)

RTRL computes the recurrent parts of the gradients during the forward pass already. Thus, activations don't have to be stored along the whole sequence. Consequently, the complexity of RTRL is independent of the sequence length T. So it can be a good alternative for a very long sequences. 