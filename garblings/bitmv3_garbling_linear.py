from enum import Enum
from random import randint
from typing import List, Optional, Tuple
from utils import Gates, is_prime, safe_prime, inverse, extended_gcd
from rsa import *
    
class BitVM3GarblingLinear:
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
        e = 5
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
        self.T = []
        self.S = []

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

                si0 = randint(1, self.modulus -1)
                si1 = randint(1, self.modulus -1)
                si2 = randint(1, self.modulus -1)
                si3 = randint(1, self.modulus -1)

                self.S.append([si0, si1, si2, si3])

                self.adaptors[i][0] = self.pow(self.labels[left][0], -1) * (a0-si0) % self.modulus
                self.adaptors[i][1] = self.pow(self.labels[left][1], -1) * (a1-si1) % self.modulus
                self.adaptors[i][2] = self.pow(self.labels[right][0], -1) * (b0-si2) % self.modulus
                self.adaptors[i][3] = self.pow(self.labels[right][1], -1) * (b1-si3) % self.modulus


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
    g = BitVM3GarblingLinear(10)

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



def polynomial_linear_attack():
    """
    Polynomial-based linear relation attack exploiting shared wire b.
    Wire b is used in both x = a AND b and y = b AND c.
    The linear relationship between b's labels in different contexts allows recovery.
    """
    print("\n--- Polynomial Linear Relation Attack on BitVM3 Garbling ---")
    print("Exploiting shared wire 'b' between two AND gates")
    
    # Create a circuit with shared wire b
    g = BitVM3GarblingLinear(30)  # Moderate size for computation
    a, b, c = g.add_inputs(3)
    x = g.add_and(a, b)  # First use of b
    y = g.add_and(b, c)  # Second use of b (same wire, different context)
    
    g.calculate_labels()
    
    # The key insight: wire b has labels b[0] and b[1]
    # When b flows into gate x, it uses adaptors from x
    # When b flows into gate y, it uses adaptors from y
    # Both create linear transformations of the same underlying label
    
    # Get the public parameters
    n = g.modulus
    e = g.exponents[0]  # Use first exponent
    e1 = g.exponents[1]  # Use first exponent
    e2 = g.exponents[2]  # Use first exponent
    e3 = g.exponents[3]  # Use first exponent
    e4 = g.exponents[4]  # Use first exponent
    
    print(f"\nModulus n = {n}")
    print(f"Using exponent e = {e}")
    
    # Evaluate with b=0 to see b[0] flowing through the circuit
    g.evaluate([0, 0, 0])  # a=0, b=0, c=0
    
    # Get the actual label for b[0]
    b0_label = g.labels[b][0]
    print(f"\nOriginal label b[0] = {b0_label}")

    # these adaptors are given to the evaluator
    alpha = g.adaptors[b][0] 
    beta = g.S[b][0]

    b0p_label = (alpha  * b0_label + beta) % n
    print(f"\nOriginal label bprime[0] = {b0p_label}")
    
    # c1 and c2 is what the evaluator has
    c1 =  rsa_encrypt(b0_label,e1,n)
    c2 =  rsa_encrypt(b0p_label,e,n)
    
    print(f"\nObserved ciphertexts (after exponentiation):")
    print(f"c1 = (b0_label)^{e1} mod n = {c1}")
    print(f"c2 = (b0p_label)^{e} mod n = {c2}")
    
    
    # Calculate the linear relationship coefficients
    try:
        # Create polynomials for GCD attack using the ACTUAL exponents
        # First polynomial: z^exp_b_in_x - c1_actual
        poly1 = [0] * (e1 + 1)
        poly1[e1] = 1
        poly1[0] = (-c1) % n
        poly1 = normalize_polynomial(poly1)
        
        # Second polynomial: (linear_coeff * z)^exp_b_in_y - c2_actual
        # This represents the linear relationship
        linear_poly = [beta %n, alpha % n]  # 0 + linear_coeff*z
        linear_poly_to_e = poly_exp(linear_poly, e, n)
        poly2 = linear_poly_to_e.copy()
        poly2[0] = (poly2[0] - c2) % n
        poly2 = normalize_polynomial(poly2)
        
        print(f"\nComputing GCD of polynomials...")
        # print(f"poly1: z^{e} - {c1_actual}")
        # print(f"poly2: ({linear_coeff}*z)^{exp_b_in_y} - {c2_actual}")
        
        # Compute GCD
        gcd_poly = poly_gcd(poly1, poly2, n)
        print(f"GCD = {print_polynomial(gcd_poly)}")
        
        # Check if GCD reveals information
        if degree(gcd_poly) == 1:
            # Extract root
            a_coeff = gcd_poly[1]
            b_coeff = gcd_poly[0]
            try:
                a_inv = mod_inverse(a_coeff, n)
                root = (-b_coeff * a_inv) % n
                print(f"\nGCD is linear! Recovered root = {root}")
                print(f"Expected b0_transformed_x = {b0_label}")
                print(f"Match? {root == b0_label}")
                
                # Now recover the original b[0] label
                # if root == b0_label:
                    # b0_transformed_x = b0 * alpha_x
                    # So b0 = b0_transformed_x / alpha_x
                    # b0_recovered = (root * mod_inverse(alpha_x, n)) % n
                    # print(f"\nRecovered b[0] label = {b0_recovered}")
                    # print(f"Original b[0] label = {b0_label}")
                    # print(f"Attack successful? {b0_recovered == b0_label}")
                    
                    # if b0_recovered == b0_label:
                    #     print("\n🎯 Successfully recovered the secret label b[0]!")
                    #     print("This demonstrates how sharing wires between gates")
                    #     print("creates exploitable linear relationships.")
            except ValueError as e:
                print(f"\nCannot extract root: {e}")
        else:
            print(f"\nGCD has degree {degree(gcd_poly)}, not linear")
            
    except ValueError as e:
        print(f"\nError in attack: {e}")
    
    print("\n" + "="*50)
    print("Summary: The attack exploits that wire b is shared between gates.")
    print("Different adaptors create linearly related transformations of b[0],")
    print("allowing polynomial GCD to recover the original label.")
    


# Main function to run the examples
if __name__ == "__main__":
    print("=== BitVM3 Garbling Linear Relation Attack Demo ===")
    print("\nThis demonstrates how the linearity relation between garbled values")
    print("induces an attack when the linear coefficients are known.")
    print("\nIn this context:")
    print("- Adaptors work as alpha (multiplicative coefficient)")
    print("- si values work as beta (additive coefficient)")
    print("- Labels are the 'messages' we want to recover")
    
    # Run the polynomial attack similar to RSA example
    polynomial_linear_attack()
    
