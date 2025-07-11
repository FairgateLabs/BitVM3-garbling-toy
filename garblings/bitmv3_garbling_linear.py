
from enum import Enum
from random import randint
from typing import List, Optional, Tuple
from garblings.utils import Gates, is_prime, safe_prime, inverse, extended_gcd
from garblings.bitmv3_garbling import BitVM3Garbling
from garblings.rsa import poly_exp, poly_gcd, rsa_encrypt, rsa_decrypt, normalize_polynomial, print_polynomial, degree, mod_inverse

class LinearAdaptorsGarbling(BitVM3Garbling):
    def __init__(self, size: int = 100):
        super().__init__(size)

    def calculate_labels(self):
        self.labels = []
        self.adaptors = []

        for i in range(len(self.gates)):
            self.labels.append([-1, -1])
            self.adaptors.append([(1,0),(1,0),(1,0),(1,0)])
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
                else:
                    T0_1 = randint(1, self.modulus) # -1 ?
                    T0_2 = (self.modulus + a0 - self.labels[left][0] * T0_1) % self.modulus

                    T1_1 = randint(1, self.modulus) # -1 ?
                    T1_2 = (self.modulus + a1 - self.labels[left][1] * T1_1) % self.modulus

                    self.adaptors[i][0] = (T0_1, T0_2)
                    self.adaptors[i][1] = (T1_1, T1_2)

                if self.labels[right][0] == -1:
                    self.labels[right][0] = b0
                    self.labels[right][1] = b1
                else:
                    T0_1 = randint(1, self.modulus) # -1 ?
                    T0_2 = (self.modulus + b0 - self.labels[right][0] * T0_1) % self.modulus

                    T1_1 = randint(1, self.modulus) # -1 ?
                    T1_2 = (self.modulus + b1 - self.labels[right][1] * T1_1) % self.modulus

                    self.adaptors[i][0] = (T0_1, T0_2)
                    self.adaptors[i][1] = (T1_1, T1_2)


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
                wire_left = self.adaptors[i][plain_left][1] + self.wires[left] * self.adaptors[i][plain_left][0] % self.modulus
                wire_right = self.adaptors[i][2+plain_right][1] + self.wires[right] * self.adaptors[i][2+plain_right][0] % self.modulus

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
    g = LinearAdaptorsGarbling(20)
    a,b,c = g.add_inputs(3)
    x = g.add_and(b, a)
    y = g.add_and(b, c)

    print(f'a = {a}'  )
    print(f'b = {b}'  )
    print(f'c = {c}'  )
    print(f'x = {x}'  )
    print(f'y = {y}'  )
    


    g.calculate_labels()
    g.evaluate([0,0,0])
    
    print("gabled")
    print("public exponents:", g.exponents)
    print("adaptors:", g.adaptors)
    print("wires:", g.wires)

    n =g.modulus
    e = g. exponents[0]
    e1 = g. exponents[1]
    e2 = g. exponents[2]
    e3 = g. exponents[3]
    e4 = g. exponents[4]

    a0 = g.wires[a]
    a_minus_e3 = g.pow(a0, -e3)
    x0 = g.wires[x]
    b1p_e = (a_minus_e3 * x0) %  n

    print(f"b1p_e = {b1p_e}"  )

    c0 = g.wires[c]
    c_minus_e3 = g.pow(c0, -e3)
    y0 = g.wires[y]
    b1_e = c_minus_e3 * y0 %  n
    print(f"b1_e = {b1_e}"  )

    b1_from_labels = g.labels[1][1]
    print(f"b1_from_labels = {b1_from_labels}"  )

    
    alpha = g.adaptors[x][1][0]
    beta = g.adaptors[x][1][1]
    b1p_from_labels = (alpha * b1_from_labels + beta) % n
    b1p_from_labels_e = pow(b1p_from_labels,e,n)



    poly1 = [0] * (e + 1)
    poly1[e] = 1
    poly1[0] = -b1_e % n  # Constant term
    poly1 = normalize_polynomial(poly1)
    print(print_polynomial(poly1))
    # print(f"p1(z) = z^{e} - {b1_e} )
    
    # Second polynomial: (alpha*z + beta)^e - c2
    linear_poly = [beta % n, alpha % n]  # beta + alpha*z
    linear_poly_to_e = poly_exp(linear_poly, e, n)
    
    # Then subtract c2
    poly2 = linear_poly_to_e.copy()
    poly2[0] = (poly2[0] - b1p_e) % n
    poly2 = normalize_polynomial(poly2)

    
    # Compute GCD
    print(f"\nComputing GCD of p1(z) and p2(z) in Z_{n}[z]:")
    try:
        gcd_poly = poly_gcd(poly1, poly2, n)
        print(f"GCD = {print_polynomial(gcd_poly)}")
        
        # Check if GCD is linear (degree 1)
        if degree(gcd_poly) == 1:
            # Extract the root: if GCD is az + b, then root is -b/a mod n
            a = gcd_poly[1]
            b = gcd_poly[0]
            try:
                a_inv = mod_inverse(a, n)
                root = (-b * a_inv) % n
                print(f"\nGCD is linear! Root = {root}")
                print(f"Verification: root = b1_from_labels? {root == b1_from_labels}")
                return root == b1_from_labels
            except ValueError:
                print(f"\nCannot find root: coefficient {a} is not invertible mod {n}")
        else:
            print(f"\nGCD has degree {degree(gcd_poly)}, not linear")
    except ValueError as e:
        print(f"Error computing GCD: {e}")
