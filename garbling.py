from enum import Enum
from random import randint
from typing import List, Optional


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

        if i > 1000000 or i * i > n:
            break

    for i in [2,3,5,7,11,13]:
        if i >= n:
            break

        if pow(n, (i - 1), i) != 1:
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


class Gates(Enum):
    INPUT = 0
    AND = 1


class Garbling:
    def __init__(self, size: int = 100):
        p = safe_prime(randint(2**(size - 1), 2**size))
        q = safe_prime(randint(2**(size - 1), 2**size))
        P = 2*p + 1
        Q = 2*q + 1

        modulus = P * Q

        self.modulus = modulus
        self.p = p
        self.q = q
        self.P = P
        self.Q = Q

        self.phi = (p - 1) * (q - 1)
        self.PHI = (P - 1) * (Q - 1)
        self.pq = p * q

        self.labels = []
        self.adaptors = []
        self.gates = []
        self.plain = []
        self.wires = []
        
        self.calculate_exponents()


    def calculate_exponents(self):
        e = 3
       

        while True:
            exponents = []
            inverses = []

            while len(exponents) < 5:
                if not is_prime(e):
                    e += 2
                    continue

                d = self.pq_pow(e, -1)
                if (e*d) % self.pq != 1:
                    print("?")
                    e += 2
                else:
                    exponents.append(e)
                    inverses.append(d)
                    e += 2

        
            h = (exponents[1]*exponents[4]*inverses[2] - exponents[3] + 4*self.PHI) % self.PHI
            ih = self.pq_pow(h, -1)  
            if (h * ih) % self.PHI == 1:
                break

        self.exponents = exponents
        self.inverses = inverses
        self.h = h
        self.invh = ih

    def add_input(self, n = 1) -> List[int]:
        ret = []

        for i in range(n):
            ret.append(len(self.gates))
            self.gates.append((Gates.INPUT,))

        return ret
    

    def add_and(self, left: int, right: int) -> int:
        gate_index = len(self.gates)
        self.gates.append((Gates.AND, left, right))
        return gate_index

    def calculate_labels(self):
        self.labels = []
        self.adaptors = []

        for i in range(len(self.gates)):
            self.labels.append([-1, -1])
            self.adaptors.append([1,1,1,1])
            self.plain.append(0)
            self.wires.append(0)


        for i in range(len(self.gates)-1, -1, -1):
            gate = self.gates[i]
            labels = self.labels[i] 
            
            if labels[0] == -1:
                labels[0] = randint(0, self.modulus - 1)
                labels[1] = randint(0, self.modulus - 1)

            if gate[0] == Gates.AND:
                b0 = labels[1] * self.pow(labels[0], -1) % self.modulus
                b0 = self.pow(b0, self.invh) % self.modulus
                
                a0 = labels[0] * self.pow(b0, -self.exponents[1]) % self.modulus
                a0 = self.pow(a0, self.inverses[0])

                a1 = labels[0] * self.pow(b0, -self.exponents[3]) % self.modulus
                a1 = self.pow(a1, self.inverses[0])

                b1 = labels[0] * self.pow(a0, -self.exponents[0]) % self.modulus
                b1 = self.pow(b1, self.inverses[2])
                
                left = gate[1]
                right = gate[2]

                if self.labels[left][0] == -1:
                    self.labels[left][0] = a0
                    self.labels[left][1] = a1

                if self.labels[right][0] == -1:
                    self.labels[right][0] = b0
                    self.labels[right][1] = b1

                self.adaptors[i][0] = self.pow(self.labels[left][0], -1) * a0 % self.modulus
                self.adaptors[i][1] = self.pow(self.labels[left][1], -1) * a1 % self.modulus
                self.adaptors[i][2] = self.pow(self.labels[right][0], -1) * b0 % self.modulus
                self.adaptors[i][3] = self.pow(self.labels[right][1], -1) * b1 % self.modulus


    def pow(self, base: int, exponent: int) -> int:
        if exponent < 0:
            exponent = self.PHI + exponent
        elif exponent == 0:
            return 1
        
        return pow(base, exponent, self.modulus)

    def pq_pow(self, base: int, exponent: int) -> int:
        if exponent < 0:
            exponent = self.phi  + exponent
        elif exponent == 0:
            return 1
        
        return pow(base, exponent, self.pq)



    def evaluate(self, inputs: List[int]): 
       index=0

       for i in range(len(self.gates)):
            gate = self.gates[i]
            if gate[0] == Gates.INPUT:
                self.plain[i] = inputs[index]
                index += 1
                self.wires[i] = self.labels[i][self.plain[i]]

            elif gate[0] == Gates.AND:
                left = gate[1]
                right = gate[2]
                plain_left = self.plain[left]
                plain_right = self.plain[right]
                wire_left = self.wires[left] * self.adaptors[i][plain_left] % self.modulus
                wire_right = self.wires[right] * self.adaptors[i][2+plain_right] % self.modulus

                wire_left = self.pow(wire_left, self.exponents[0])
                wire_right = self.pow(wire_right, self.exponents[1+plain_left*2 + plain_right])

                self.plain[i] = (plain_left * plain_right) 
                self.wires[i] = (wire_left * wire_right) % self.modulus 

    


g = Garbling(10)

g.add_input(2)
x = g.add_and(0, 1)
#y = g.add_and(1, 0)
g.calculate_labels()

for i in range(4):
    i = 2
    inputs = [i // 2, i % 2 ]
    g.evaluate(inputs)
    valid = []
    for j in range(len(g.gates)):
        valid.append(g.wires[j] == g.labels[j][g.plain[j]])

    print(g.wires, valid)
    

