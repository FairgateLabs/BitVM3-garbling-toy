# RSA-Based Garbling: Toy Implementation and Attack Demo

This project provides toy implementations of two RSA-based garbling schemes, along with a minimal attack that demonstrates how a malicious evaluator can forge additional wire labels.

## Garbling Schemes

The code includes implementations of three garbling schemes based on the following two papers:

- **BitVM3: Efficient Computation on Bitcoin**  
  Source: [bitvm.org/bitvm3.pdf](https://bitvm.org/bitvm3.pdf)

- **Label Forward Propagation: Instantiating BitVM3**  
  Source: [goat.network/bitvm3-label-forward-propagation](https://www.goat.network/bitvm3-label-forward-propagation)

And from a suggestion to use affine functions for adaptors proposed on the BitVM builders telegram group.

## Attack Overview

A minimal example demonstrates how a malicious evaluator can exploit the scheme. The attack uses a small circuit consisting of two AND gates and three inputs.  
Given the public data, circuit adaptors, and wire labels for the input `[0, 0, 0]`, the evaluator is able to **forge at least one additional wire label**.

## Detailed Explanation

For a full explanation of the attack strategy, see the [detailed write-up](https://fairgate.io/files/bitvm3-sec.pdf)

## Running the Demo

Clone the repository and run the demo script. No additional dependencies are required.

```bash
python demo.py
