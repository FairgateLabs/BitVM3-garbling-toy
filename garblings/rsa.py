# Polynomial representation using array of coefficients
# polynomial[0] = constant term
# polynomial[1] = coefficient of x
# polynomial[2] = coefficient of x^2
# and so on...

# Example: 3x^2 + 2x + 5
# Represented as [5, 2, 3]
polynomial = [5, 2, 3]

def evaluate_polynomial(poly, x):
    """Evaluate polynomial at given x value"""
    result = 0
    for i, coeff in enumerate(poly):
        result += coeff * (x ** i)
    return result

def reduce_polynomial_mod_n(poly, n):
    """Reduce polynomial coefficients modulo n"""
    return [coeff % n for coeff in poly]

def mod_inverse(a, n):
    """Find modular inverse of a modulo n using extended Euclidean algorithm"""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    gcd, x, _ = extended_gcd(a % n, n)
    if gcd != 1:
        raise ValueError(f"Modular inverse does not exist for {a} mod {n}")
    return (x % n + n) % n

def degree(poly):
    """Return the degree of the polynomial"""
    for i in range(len(poly) - 1, -1, -1):
        if poly[i] != 0:
            return i
    return -1

def normalize_polynomial(poly):
    """Remove leading zeros from polynomial"""
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    if not poly:
        return [0]
    return poly

def poly_divmod(dividend, divisor, n):
    """Divide two polynomials in Z_n[X] and return quotient and remainder"""
    dividend = dividend.copy()
    divisor = divisor.copy()
    
    normalize_polynomial(dividend)
    normalize_polynomial(divisor)
    
    if divisor == [0]:
        raise ValueError("Division by zero polynomial")
    
    quotient = []
    
    while degree(dividend) >= degree(divisor):
        # Get leading coefficients
        dividend_lead = dividend[degree(dividend)]
        divisor_lead = divisor[degree(divisor)]
        
        # Compute coefficient for quotient term
        try:
            coeff = (dividend_lead * mod_inverse(divisor_lead, n)) % n
        except ValueError:
            raise ValueError(f"Cannot divide: leading coefficient {divisor_lead} is not invertible mod {n}")
        
        quotient.append(coeff)
        
        # Subtract divisor * coeff from dividend
        deg_diff = degree(dividend) - degree(divisor)
        for i in range(len(divisor)):
            if i + deg_diff < len(dividend):
                dividend[i + deg_diff] = (dividend[i + deg_diff] - coeff * divisor[i]) % n
        
        normalize_polynomial(dividend)
    
    quotient.reverse()
    remainder = dividend
    
    return normalize_polynomial(quotient), normalize_polynomial(remainder)

def poly_mul(a, b, n):
    """Multiply two polynomials in Z_n[X]"""
    if not a or not b:
        return [0]
    
    result = [0] * (len(a) + len(b) - 1)
    
    for i in range(len(a)):
        for j in range(len(b)):
            result[i + j] = (result[i + j] + a[i] * b[j]) % n
    
    return normalize_polynomial(result)

def poly_mod(poly, modulus, n):
    """Reduce polynomial modulo another polynomial in Z_n[X]"""
    _, remainder = poly_divmod(poly, modulus, n)
    return remainder

def poly_exp(base, exponent, n, modulus=None):
    """
    Compute base^exponent in Z_n[X], optionally modulo a polynomial modulus.
    Uses binary exponentiation for efficiency.
    
    Args:
        base: polynomial to exponentiate
        exponent: non-negative integer exponent
        n: modulus for coefficients (Z_n)
        modulus: optional polynomial modulus
    
    Returns:
        base^exponent mod modulus in Z_n[X]
    """
    if exponent < 0:
        raise ValueError("Exponent must be non-negative")
    
    if exponent == 0:
        return [1]
    
    # Copy and normalize base
    base = [coeff % n for coeff in base.copy()]
    normalize_polynomial(base)
    
    # Binary exponentiation
    result = [1]  # Start with polynomial 1
    
    while exponent > 0:
        if exponent % 2 == 1:
            result = poly_mul(result, base, n)
            if modulus is not None:
                result = poly_mod(result, modulus, n)
        
        base = poly_mul(base, base, n)
        if modulus is not None:
            base = poly_mod(base, modulus, n)
        
        exponent //= 2
    
    return normalize_polynomial(result)

def poly_gcd(a, b, n):
    """Compute GCD of two polynomials in Z_n[X] using Euclidean algorithm"""
    a = [coeff % n for coeff in a.copy()]
    b = [coeff % n for coeff in b.copy()]
    
    normalize_polynomial(a)
    normalize_polynomial(b)
    
    while b != [0]:
        try:
            _, remainder = poly_divmod(a, b, n)
            a, b = b, remainder
        except ValueError as e:
            # If division fails due to non-invertible leading coefficient
            # we cannot compute GCD in this ring
            raise ValueError(f"Cannot compute GCD in Z_{n}[X]: {e}")
    
    # Make the GCD monic (leading coefficient = 1) if possible
    if a and a[degree(a)] != 0:
        try:
            lead_inv = mod_inverse(a[degree(a)], n)
            a = [(coeff * lead_inv) % n for coeff in a]
        except ValueError:
            # Cannot make monic, return as is
            pass
    
    return normalize_polynomial(a)

# RSA Functions
def gcd(a, b):
    """Compute GCD of two integers"""
    while b:
        a, b = b, a % b
    return a

def rsa_keygen(p, q, e=None):
    """
    Generate RSA keys from primes p and q.
    
    Args:
        p: first prime
        q: second prime
        e: public exponent (default: 65537 or smallest valid e)
    
    Returns:
        dict with keys: n, e, d, p, q, phi
    """
    n = p * q
    phi = (p - 1) * (q - 1)
    
    # Choose e if not provided
    if e is None:
        e = 65537
        if gcd(e, phi) != 1:
            e = 3
            while gcd(e, phi) != 1:
                e += 2
    
    # Verify e is valid
    if gcd(e, phi) != 1:
        raise ValueError(f"e={e} is not coprime with phi(n)={phi}")
    
    # Compute d = e^(-1) mod phi
    d = mod_inverse(e, phi)
    
    return {
        'n': n,
        'e': e,
        'd': d,
        'p': p,
        'q': q,
        'phi': phi
    }

def rsa_encrypt(message, e, n):
    """
    Encrypt a message using RSA.
    
    Args:
        message: integer message (0 <= message < n)
        e: public exponent
        n: modulus
    
    Returns:
        ciphertext (integer)
    """
    if not (0 <= message < n):
        raise ValueError(f"Message must be in range [0, {n-1}]")
    
    return pow(message, e, n)

def rsa_decrypt(ciphertext, d, n):
    """
    Decrypt a ciphertext using RSA.
    
    Args:
        ciphertext: encrypted message (integer)
        d: private exponent
        n: modulus
    
    Returns:
        plaintext (integer)
    """
    return pow(ciphertext, d, n)

def print_polynomial(poly):
    """Print polynomial in readable format"""
    terms = []
    for i, coeff in enumerate(poly):
        if coeff != 0:
            if i == 0:
                terms.append(str(coeff))
            elif i == 1:
                if coeff == 1:
                    terms.append("x")
                elif coeff == -1:
                    terms.append("-x")
                else:
                    terms.append(f"{coeff}x")
            else:
                if coeff == 1:
                    terms.append(f"x^{i}")
                elif coeff == -1:
                    terms.append(f"-x^{i}")
                else:
                    terms.append(f"{coeff}x^{i}")
    
    # Reverse to show highest degree first
    terms.reverse()
    
    # Join with proper signs
    if not terms:
        return "0"
    
    result = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            result += f" {term}"
        else:
            result += f" + {term}"
    
    return result

# Example usage
if __name__ == "__main__":
    # Define polynomial: 3x^2 + 2x + 5
    poly1 = [5, 2, 3]
    print(f"Polynomial: {print_polynomial(poly1)}")
    print(f"Value at x=2: {evaluate_polynomial(poly1, 2)}")
    
    # Another example: -x^3 + 4x^2 - 3x + 7
    poly2 = [7, -3, 4, -1]
    print(f"\nPolynomial: {print_polynomial(poly2)}")
    print(f"Value at x=1: {evaluate_polynomial(poly2, 1)}")
    
    # Example of reducing polynomial modulo N
    print("\n--- Polynomial Reduction mod N ---")
    poly3 = [5, 3, 2]
    n = 3
    reduced_poly = reduce_polynomial_mod_n(poly3, n)
    print(f"Original polynomial: {poly3} = {print_polynomial(poly3)}")
    print(f"Reduced mod {n}: {reduced_poly} = {print_polynomial(reduced_poly)}")
    
    # Example of polynomial GCD in Z_n[X]
    print("\n--- Polynomial GCD in Z_n[X] ---")
    # Example in Z_5[X]
    n = 5
    # a(x) = x^3 + 2x + 1
    a = [1, 2, 0, 1]
    # b(x) = x^2 + 1
    b = [1, 0, 1]
    
    print(f"Computing GCD in Z_{n}[X]:")
    print(f"a(x) = {print_polynomial(a)}")
    print(f"b(x) = {print_polynomial(b)}")
    
    try:
        gcd_result = poly_gcd(a, b, n)
        print(f"GCD = {print_polynomial(gcd_result)}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Another example with common factor
    print("\n--- Example with common factor ---")
    n = 7
    # a(x) = (x+1)(x+2) = x^2 + 3x + 2
    a = [2, 3, 1]
    # b(x) = (x+1)(x+3) = x^2 + 4x + 3
    b = [3, 4, 1]
    
    print(f"Computing GCD in Z_{n}[X]:")
    print(f"a(x) = {print_polynomial(a)}")
    print(f"b(x) = {print_polynomial(b)}")
    
    try:
        gcd_result = poly_gcd(a, b, n)
        print(f"GCD = {print_polynomial(gcd_result)}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example of polynomial exponentiation
    print("\n--- Polynomial Exponentiation ---")
    n = 7
    # base(x) = x + 2
    base = [2, 1]
    exp = 3
    
    print(f"Computing ({print_polynomial(base)})^{exp} in Z_{n}[X]:")
    result = poly_exp(base, exp, n)
    print(f"Result = {print_polynomial(result)}")
    
    # Verify: (x+2)^3 = x^3 + 6x^2 + 12x + 8
    print(f"Coefficients: {result}")
    
    # Example with modulus polynomial
    print("\n--- Polynomial Exponentiation with Modulus ---")
    n = 5
    base = [1, 1]  # x + 1
    exp = 10
    modulus = [1, 0, 1]  # x^2 + 1
    
    print(f"Computing ({print_polynomial(base)})^{exp} mod ({print_polynomial(modulus)}) in Z_{n}[X]:")
    result = poly_exp(base, exp, n, modulus)
    print(f"Result = {print_polynomial(result)}")
    
    # Another example
    n = 11
    base = [2, 3]  # 3x + 2
    exp = 5
    modulus = [1, 0, 0, 1]  # x^3 + 1
    
    print(f"\nComputing ({print_polynomial(base)})^{exp} mod ({print_polynomial(modulus)}) in Z_{n}[X]:")
    result = poly_exp(base, exp, n, modulus)
    print(f"Result = {print_polynomial(result)}")
    
    # RSA Example
    print("\n--- RSA Example ---")
    
    # Small primes for demonstration
    p = 11
    q = 13
    e = 7
    
    print(f"Using primes p={p}, q={q}")
    
    # Generate RSA keys
    keys = rsa_keygen(p, q, e)
    print(f"\nRSA Key Generation:")
    print(f"n = p * q = {keys['n']}")
    print(f"phi(n) = (p-1)(q-1) = {keys['phi']}")
    print(f"Public exponent e = {keys['e']}")
    print(f"Private exponent d = {keys['d']}")
    
    # Verify ed ≡ 1 (mod phi(n))
    print(f"\nVerification: e*d mod phi(n) = {(keys['e'] * keys['d']) % keys['phi']}")
    
    # Encrypt and decrypt a message
    message = 6
    print(f"\nOriginal message: {message}")
    
    # Encrypt
    ciphertext = rsa_encrypt(message, keys['e'], keys['n'])
    print(f"Encrypted: {ciphertext}")
    
    # Decrypt
    decrypted = rsa_decrypt(ciphertext, keys['d'], keys['n'])
    print(f"Decrypted: {decrypted}")
    
    print(f"\nDecryption successful: {decrypted == message}")
    
    # Another example with custom e
    print("\n--- RSA with custom e ---")
    p2 = 17
    q2 = 19
    e2 = 5
    
    print(f"Using p={p2}, q={q2}, e={e2}")
    keys2 = rsa_keygen(p2, q2, e2)
    
    print(f"n = {keys2['n']}")
    print(f"d = {keys2['d']}")
    
    message2 = 100
    cipher2 = rsa_encrypt(message2, keys2['e'], keys2['n'])
    plain2 = rsa_decrypt(cipher2, keys2['d'], keys2['n'])
    
    print(f"\nMessage: {message2} -> Cipher: {cipher2} -> Decrypted: {plain2}")
    
    # Linear Related Messages Attack
    print("\n--- Linear Related Messages Attack ---")
    
    # Use the existing RSA keys
    m = 42  # Choose a message
    e = keys['e']
    n = keys['n']
    
    # General linear transformation: m2 = alpha*m + beta
    alpha = 3
    beta = 7
    
    # Ensure alpha is coprime with n
    if gcd(alpha, n) != 1:
        print(f"Warning: alpha={alpha} is not coprime with n={n}")
    
    # Compute m2 = alpha*m + beta
    m2 = (alpha * m + beta) % n
    
    # Encrypt m and m2
    c1 = rsa_encrypt(m, e, n)
    c2 = rsa_encrypt(m2, e, n)
    
    print(f"\nm = {m}")
    print(f"alpha = {alpha}, beta = {beta}")
    print(f"m2 = alpha*m + beta = {m2}")
    print(f"e = {e}")
    print(f"n = {n}")
    print(f"\nc1 = m^e mod n = {c1}")
    print(f"c2 = m2^e mod n = {c2}")
    
    # Create polynomials z^e - c1 and (alpha*z + beta)^e - c2
    print(f"\nCreating polynomials in Z_{n}[z]:")
    
    # First polynomial: z^e - c1
    # z^e represented as coefficient array with 1 at position e
    poly1 = [0] * (e + 1)
    poly1[e] = 1
    poly1[0] = -c1 % n  # Constant term
    poly1 = normalize_polynomial(poly1)
    print(f"p1(z) = z^{e} - {c1}")
    
    # Second polynomial: (alpha*z + beta)^e - c2
    # First compute (alpha*z + beta)
    linear_poly = [beta % n, alpha % n]  # beta + alpha*z
    
    # Then compute (alpha*z + beta)^e
    linear_poly_to_e = poly_exp(linear_poly, e, n)
    
    # Then subtract c2
    poly2 = linear_poly_to_e.copy()
    poly2[0] = (poly2[0] - c2) % n
    poly2 = normalize_polynomial(poly2)
    print(f"p2(z) = ({alpha}*z + {beta})^{e} - {c2}")
    
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
                print(f"Verification: root = m? {root == m}")
            except ValueError:
                print(f"\nCannot find root: coefficient {a} is not invertible mod {n}")
        else:
            print(f"\nGCD has degree {degree(gcd_poly)}, not linear")
    except ValueError as e:
        print(f"Error computing GCD: {e}")
    
    # Try another example with different alpha, beta
    print("\n--- Another Linear Related Messages Example ---")
    
    # Different linear transformation
    alpha2 = 2
    beta2 = 5
    
    # Use smaller values for this example
    m_small = 15
    m2_small = (alpha2 * m_small + beta2) % n
    
    # Encrypt
    c1_small = rsa_encrypt(m_small, e, n)
    c2_small = rsa_encrypt(m2_small, e, n)
    
    print(f"\nm = {m_small}")
    print(f"alpha = {alpha2}, beta = {beta2}")
    print(f"m2 = alpha*m + beta = {m2_small}")
    print(f"\nc1 = m^e mod n = {c1_small}")
    print(f"c2 = m2^e mod n = {c2_small}")
    
    # Create polynomials
    poly1_small = [0] * (e + 1)
    poly1_small[e] = 1
    poly1_small[0] = -c1_small % n
    poly1_small = normalize_polynomial(poly1_small)
    
    linear_poly_small = [beta2 % n, alpha2 % n]
    linear_poly_to_e_small = poly_exp(linear_poly_small, e, n)
    poly2_small = linear_poly_to_e_small.copy()
    poly2_small[0] = (poly2_small[0] - c2_small) % n
    poly2_small = normalize_polynomial(poly2_small)
    
    print(f"\nComputing GCD...")
    try:
        gcd_poly_small = poly_gcd(poly1_small, poly2_small, n)
        print(f"GCD = {print_polynomial(gcd_poly_small)}")
        
        if degree(gcd_poly_small) == 1:
            a = gcd_poly_small[1]
            b = gcd_poly_small[0]
            try:
                a_inv = mod_inverse(a, n)
                root = (-b * a_inv) % n
                print(f"Root = {root}")
                print(f"Verification: root = m? {root == m_small}")
            except ValueError:
                print(f"Cannot find root: coefficient {a} is not invertible mod {n}")
        else:
            print(f"GCD has degree {degree(gcd_poly_small)}, not linear")
    except ValueError as e:
        print(f"Error computing GCD: {e}")
