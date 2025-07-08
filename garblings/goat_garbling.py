from enum import Enum
from random import randint
from typing import List, Optional, Tuple
from garblings.utils import Gates, is_prime, safe_prime, inverse, extended_gcd
from garblings.bitmv3_garbling import BitVM3Garbling

class GoatGarbling(BitVM3Garbling):
    def __init__(self, size: int = 100):
        super().__init__(size)
        
    def calculate_labels(self):
        self.labels = []
        self.adaptors = []


        for i in range(len(self.gates)):
            self.plain.append(0)

            gate = self.gates[i]
            labels = [0,0]
            adaptors = [1,1]

            if gate[0] == Gates.INPUT:
                labels[0] = randint(0, self.modulus - 1)
                labels[1] = randint(0, self.modulus - 1)

            elif gate[0] == Gates.AND:
                left = self.labels[gate[1]]
                right = self.labels[gate[2]]

                out0 = self.pow(left[0], self.exponents[0]) * self.pow(right[0], self.exponents[1]) % self.modulus
                out1 = self.pow(left[0], self.exponents[0]) * self.pow(right[1], self.exponents[2]) % self.modulus
                out2 = self.pow(left[1], self.exponents[0]) * self.pow(right[0], self.exponents[3]) % self.modulus
                out3 = self.pow(left[1], self.exponents[0]) * self.pow(right[1], self.exponents[4]) % self.modulus

                labels[0] = out0
                labels[1] = out3
                adaptors[0] = out0 * inverse(out1, self.modulus) % self.modulus
                adaptors[1] = out0 * inverse(out2, self.modulus) % self.modulus

            self.labels.append(labels)
            self.adaptors.append(adaptors)
            self.wires.append(0)

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
                wire_left = self.wires[left]
                wire_right = self.wires[right]

                cnt = plain_left*2 + plain_right
                wire_value = self.pow(wire_left, self.exponents[0]) * self.pow(wire_right, self.exponents[1+cnt]) % self.modulus

                if cnt == 1:
                    wire_value = wire_value * self.adaptors[i][0] % self.modulus
                elif cnt == 2:
                    wire_value = wire_value * self.adaptors[i][1] % self.modulus

                self.plain[i] = (plain_left * plain_right) 
                self.wires[i] = wire_value
            else: 
                raise ValueError(f"Unknown gate type: {gate[0]}")

    
def test():
    g = GoatGarbling(10)

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
    g = GoatGarbling(20)
    a,b,c = g.add_inputs(3)
    x = g.add_and(a, b)
    y = g.add_and(b, c)

    g.calculate_labels()
    g.evaluate([0,0,0])
    
    print("gabled")
    print("public exponents:", g.exponents)
    print("adaptors:", g.adaptors)
    print("wires:", g.wires)


    a0 = g.wires[a]
    b0 = g.wires[b]
    c0 = g.wires[c]
    x0 = g.wires[x]
    y0 = g.wires[y]

    a0_e = g.pow(a0, g.exponents[0])
    b0_e1 = g.pow(b0, g.exponents[1])
    b1_e2 = b0_e1 * inverse(g.adaptors[x][0], g.modulus) % g.modulus
    b1_e__c0_e3 = y0 * inverse(g.adaptors[y][1], g.modulus) % g.modulus

    b1_e = b1_e__c0_e3 * inverse(g.pow(c0, g.exponents[3]), g.modulus) % g.modulus
    gcd, k1, k2 = extended_gcd(g.exponents[0], g.exponents[2])
    b1 = g.pow(b1_e, k1) * g.pow(b1_e2, k2) % g.modulus
   
    print("b1_recovered = ", b1, b1 == g.labels[b][1])
    if b1 == g.labels[b][1]:
        print("recovered 1 input label")
    return b1 == g.labels[b][1]


