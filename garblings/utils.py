from enum import Enum
from random import randint
from typing import List, Optional, Tuple

def is_prime(n :int) -> bool:
    is_prime = True
        
    i = 2
    while True:
        if n % i == 0:
            is_prime = False
            break
        
        if i == 2: 
            i+=1
        else:
            i += 2

        if  i > 10000 or i * i > n:
            break

        

    for i in [2,3,5,7,11,13]:
        if i >= n:
            break

        if pow(i, (n - 1), n) != 1:
            is_prime = False
            break

        i += 1

    return is_prime

def prime(starting: int) -> int:
    """ Finding the next prime number using a probabilistic test. """

    if starting % 2 == 0:
        starting += 1

    while True:
        if is_prime(starting):
            break

        starting += 2    

    return starting

def safe_prime(starting: int) -> int:
    while True:
        starting = prime(starting)
        if is_prime(starting * 2 + 1):
            return starting

        starting += 2
        print(".",end='')



class Gates(Enum):
    INPUT = 0
    AND = 1



def inverse(x: int, modulus: int) -> Optional[int]:
        """ Returns the inverse of x modulo modulus, or None if it does not exist. """
        if x == 0:
            return None

        try:
            d = pow(x, -1, modulus)
        except ValueError:
            return None
        
        if (x * d) % modulus == 1:
            return d
        else:
            return None

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    x0, x1, y0, y1 = 1, 0, 0, 1
    while b != 0:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1

    return a, x0, y0
