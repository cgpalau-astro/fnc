"""RSA encryptation algorithm.

Example
-------
>>> private_key = rsa.PrivateKey()
>>> public_key = rsa.PublicKey(private_key.e, private_key.n)

>>> message = rsa.str_to_int('password')

>>> cm = rsa.encrypt(message, public_key)
>>> dm = rsa.decrypt(cm, private_key)

>>> message == dm
True
>>> print(rsa.int_to_str(message))
password"""

from . import core as _core

#-----------------------------------------------------------------------------

class PrivateKey:
    """Private key.

    Note
    ----
    1)  p, q are prime numbers.
    2)  e is an integer such that 1 <= e <= l and gcd(e, l) = 1 .
    3)  n determines the maximum number m that can be encrypted: 0 <= m < n.
    4)  The encrypted message cm has len(str(cm)) = n."""

    def __init__(self):
        self.p = 20777555927117689963046939110472962807913905958685261923625122686380274101694666918565197092149517802746946815602015031199667385880358835634533075531978037129553363269400478836980687459859651493372647237091804009925227379760287412107872274063314464150770782462858020056645984873018017106839963927504780639257826346599796793963489171519752444540603015885288310667473122608869792318748074220824659501659060183974905899695677534934647012110102425411387591090138927274366532661615687885814614269717622653177166592531248885785208103387026549408489539022245394803106940209212513397055588985724945196883955014122962213372873
        self.q = 26028269096130690169619972032888732374604664749524125664710951577163547183557366168518735174829847621608378501189881862413178094130499880446245189829094880707063514959591610586317364639649909025843196235911734152053170129138976994586879948520607716634733560941012073975733480018130776904154331853671696279841119235298287099840186498879636007891273517962391512599510032572066758454930530872408012265659557703729210575457695282719127285843821668305286506482662143354607422868664719680129343441021339296113145698934712871598619082909410153947434675998695397911051164972828123148203581840416001633796433245179461254321987
        self.e = 65_537
        #---------------------------------------------------------------
        self.n = self.p * self.q
        self._l = _core.lcm(self.p - 1, self.q - 1)
        self._d = _core.modular_multiplicative_inverse(self.e, self._l)
        #---------------------------------------------------------------
        self._dp = self._d % (self.p - 1)
        self._dq = self._d % (self.q - 1)
        self._qinv = _core.modular_multiplicative_inverse(self.q, self.p)

        #---------------------------------------------------------------
        def key_length(x):
            """Size of a integer number in bits"""
            return x.bit_length()

        self.key_length = f"{key_length(self.n)} bits"
        #---------------------------------------------------------------

class PublicKey:
    """Public key:

    Note
    ----
    1) e, n : int"""

    def __init__(self, e, n):
        self.e = e
        self.n = n

#-----------------------------------------------------------------------------

def str_to_int(x):
    #Pad with b'\x01' (1) to preserve trailing zeroes
    x_pad = x.encode('utf-8') + b'\x01'
    x_int = int.from_bytes(x_pad, 'little')
    return x_int

def int_to_str(x):
    x_pad = x.to_bytes((x.bit_length() + 7) // 8, 'little')
    #Remove b'\x01' (1)
    x_str = x_pad[:-1].decode('utf-8')
    return x_str

#-----------------------------------------------------------------------------

def encrypt(message, public_key):
    #return (message**e) % n
    encrypted_message = pow(message, public_key.e, public_key.n)
    return encrypted_message

def decrypt(encrypted_message, private_key):
    #pow(encrypted_message, private_key.d, private_key.n)
    p = private_key.p
    q = private_key.q
    m1 = pow(encrypted_message, private_key._dp, p)
    m2 = pow(encrypted_message, private_key._dq, q)
    h = (private_key._qinv * (m1 - m2)) % p
    message = (m2 + h * q) % (p * q)
    return message

#-----------------------------------------------------------------------------
