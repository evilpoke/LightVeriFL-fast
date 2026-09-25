# LightVeriFL-fast

This repo is to implement the LigthVeriFL scheme that enables lightweight verification of the aggregation result provided by the server at each iteration in federated learning.
For fast elliptic curve cryptography, we use the open-source fastecdsa Python library (https://github.com/AntonKueltz/fastecdsa). Please follow the instructions on https://github.com/AntonKueltz/fastecdsa
for the installation of fastecdsa.

In order to run the experiments, in the ServerVerification folder, execute

- mpirun -n {N+1} python LightVeriFL_EC_fastecdsa_amortized.py {N} {N-D} {d} {L}.
Here, N is the number of users, D is the number of dropped users, d is the model dimension, and L is the batch size for the amortization.

Stefan Schärdinger progress:

UNDERSTANDING:

- using kerasdefault conda, but does not use keras as lib. 
- does not actually do the gradient decent but uses random things


CODE DEBUGGING:

- debug for vscode injected
- in another terminal do:
  - mpirun -n 4 python LightVeriFL_EC_fastecdsa_amortized.py 4 3 1 5
- and then do "Attach to MPI rank 1"

I get:
mpi4py.MPI.Exception: Invalid rank, error stack:
internal_Send_c(312): MPI_Send_c(buf=0x57edad70b910, count=3, MPI_LONG, 4, 0, MPI_COMM_WORLD) failed
internal_Send_c(232): Invalid rank has value 4 but must be nonnegative and less than 4


so now i do:
mpirun -n 5 python LightVeriFL_EC_fastecdsa_amortized.py 4 3 10000 5

I get:
Verification/utils/function.py", line 52, in PI
    tmp = v % p  # np.mod(v, p)
          ~~^~~
OverflowError: Python int too large to convert to C long 


I now analyse clientside: I set the port in the python to 80 such that i catch rank 1 (of a client)


Apparenly, gen_Lagrange_coeffs just fucks up.

I now want to test gen_Lagrange_coeffs, in particular: I would really like to know in detail how W is constructed (also for LightSecAgg)
Hence: I now to back to LightSecAgg, because here the code does not work.

Update from LightSecAgg. I now know how it's gen_lagrange_coeffs works and apparently gen_lagrange_coeffs is called similarly. Their codebase has huge problems with correct setup. I would like to use FedML anyway to do actual learning stuff, sooo....

I now integrated the stuff from LightSegAgg into here (so the FedML codebase)
Right now i get:

ValueError: cannot reshape array of size 13083 into shape (5,2616)



