from enum import Enum
from random import randint
from typing import List, Optional, Tuple
from garblings.utils import Gates, is_prime, safe_prime, inverse, extended_gcd
    
class BitVM3Garbling:
    def __init__(self, size: int = 100):
        p = safe_prime(randint(2**(size - 1), 2**size))
        while True:
            q = safe_prime(randint(2**(size - 1), 2**size))
            if p != q:
                break

        P = 2*p + 1
        Q = 2*q + 1

        modulus = P * Q

        self.modulus = modulus
        self.p = p
        self.q = q
        self.P = P
        self.Q = Q

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
        exponents = []
        inverses = []
        while True:
            while len(exponents) < 5:
                if not is_prime(e):
                    e += 2
                    continue

                d = inverse(e, self.PHI)

                if d is None:
                    print("?")
                    e += 2
                    continue
                else:
                    exponents.append(e)
                    inverses.append(d)
                    e += 2

        
            h = (exponents[1]*exponents[4]*inverses[2] - exponents[3]) % self.PHI
            ih = inverse(h, self.pq)  
            if ih is not None:
                break

            exponents = exponents[0:4]
            inverses = inverses[0:4]

        self.exponents = exponents
        self.inverses = inverses
        self.h = h
        self.invh = ih


    def add_inputs(self, n = 1) -> List[int]:
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
                labels[0] = self.pow(randint(0, self.modulus - 1),4)
                labels[1] = self.pow(randint(0, self.modulus - 1),4)

            if gate[0] == Gates.AND:
                label0i = inverse(labels[0], self.modulus)

                b0 = labels[1] * label0i % self.modulus
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
            else: 
                raise ValueError(f"Unknown gate type: {gate[0]}")

    


def test():
    g = BitVM3Garbling(10)

    a,b,c = g.add_inputs(3)
    x = g.add_and(a, b)
    y = g.add_and(b, c)
    z1 = g.add_and(x, y)
    z2 = g.add_and(x, y)
    g.add_and(z1, z2)

    g.calculate_labels()

    print(g.exponents)
    correct = True

    for i in range(8):
        inputs = [ int(x) for x in bin(i)[2:].zfill(3) ]
      
        g.evaluate(inputs)
        valid = []
       
        for j in range(len(g.gates)):
            valid.append(g.wires[j] == g.labels[j][g.plain[j]])
            if False in valid:
                correct = False
                print(f"Gate {j} failed: {g.wires[j]} != {g.labels[j][g.plain[j]]}")

        print(valid)
        
    if correct:
        print("All gates evaluated correctly.")
    else:
        print("Some gates failed to evaluate correctly.")


def attack():
    g = BitVM3Garbling(20)
    a,b,c = g.add_inputs(3)
    x = g.add_and(a, b)
    y = g.add_and(b, c)

    g.calculate_labels()
    g.evaluate([0,0,0])
    
    print("gabled")
    print("public exponents:", g.exponents)
    print("adaptors:", g.adaptors)
    print("wires:", g.wires)


    b0 = g.wires[b]
    x0 = g.wires[x]
    b0p = b0 * g.adaptors[x][2] % g.modulus
    c0 = g.wires[c]

    y0 = g.wires[y]

    b0_e1 = g.pow(b0p, g.exponents[1])
    adaptor_minus_e2 = g.pow(g.adaptors[x][3], -g.exponents[2])
    b1_e2 = b0_e1 * adaptor_minus_e2 % g.modulus

    
    c0_e3 = g.pow(c0, g.exponents[3])
    b1_e = y0 * g.pow(c0_e3, -1) % g.modulus
    
    b1 = g.labels[b][1] 
    b1p = b1 * g.adaptors[x][3] % g.modulus

    gcd, k1, k2 = extended_gcd(g.exponents[0], g.exponents[2])
    print("gcd(e1,e3):", gcd, "k1:", k1, "k2:", k2)

    print(b1_e, b1_e2)
    
    b1_recovered = g.pow(b1_e, k1) *  g.pow(b1_e2, k2) % g.modulus
    b1p_recovered = b1_recovered * g.adaptors[x][3] % g.modulus
    print("b1_recovered:", b1_recovered, b1_recovered==b1)
    print("b1p_recovered:", b1p_recovered, b1p_recovered==b1p)

    x1_recovered = g.pow(b0p, -g.exponents[3]) * g.pow(b1p_recovered, g.exponents[4]) * x0 % g.modulus  
    print("x1 recovered:", x1_recovered, x1_recovered == g.labels[x][1])
    

    if b1p_recovered == b1p and x1_recovered == g.labels[x][1]:
        print("recovered 1 input label and 1 output label")

    return b1_recovered == b1
