# Attack DPA Contest v2 with CPA and DPA

In this guide, we will show how to attack the DPA V2 trace set with CPA and DPA method, with the help of the `scared` library.
You can also launch a tutorial in [Binder from this repository](https://mybinder.org/v2/gl/eshard%2Fscared-notebooks/master).

## The DPA V2 trace set

For this example, we will use the [DPA Contest v2](http://www.dpacontest.org/v2/download.php). In the second version, they provide different sets of power measurement traces acquired from the [SASEBO-GII](http://satoh.cs.uec.ac.jp/SASEBO/en/board/sasebo-g2.html) platform performing AES-128 encryptions.

In particular, you must download the raw traces of the public database DPA_contest2_public_base_diff_vcc_a128_2009_12_23.tar.bz2 archive.

### Download and extract traces data

If you want to download them, execute the following command lines. <br />
⚠ **Warning:** **3.5**GB split in 4 files.

```python
!wget -O dpa_v2.tar.bz2.part0 http://www.dpacontest.org/v2/data/traces/DPA_contest2_public_base_diff_vcc_a128_2009_12_23.tar.bz2.part0
!wget -O dpa_v2.tar.bz2.part1 http://www.dpacontest.org/v2/data/traces/DPA_contest2_public_base_diff_vcc_a128_2009_12_23.tar.bz2.part1
!wget -O dpa_v2.tar.bz2.part2 http://www.dpacontest.org/v2/data/traces/DPA_contest2_public_base_diff_vcc_a128_2009_12_23.tar.bz2.part2
!wget -O dpa_v2.tar.bz2.part3 http://www.dpacontest.org/v2/data/traces/DPA_contest2_public_base_diff_vcc_a128_2009_12_23.tar.bz2.part3
```

Once downloaded, the 4 archive parts must be merged as follow:

```python
# Merge archive parts
!cat dpa_v2.tar.bz2.part{0..3} > dpa_v2.tar.bz2
```

We can then extract the archive:

```python
!tar -xjf dpa_v2.tar.bz2
```

The extraction process can take long.
The traces will be extracted into the `DPA_contest2_public_base_diff_vcc_a128_2009_12_23` directory (created during extraction).

### Convert traces

After extracting the archive of the public base, we obtained 640,000 '.csv' traces files that correspond to 32 subsets of 20,000 traces.
What changes among those 32 subsets are the key used for encryption.
For each trace, a random plaintext was generated for encryption.

Here is the kind of file we have:

`./DPA_contest2_public_base_diff_vcc_a128_2009_12_23/wave_aist-aes-agilent_2009-12-31_01-33-21_n=29479_k=13198a2e03707344a4093822299f31d0_m=8d4207d890f562309724374abde6e569_c=1ed272974f92997d97e2b4f680fee68c.csv`

Each trace file has a constant sized header of 24 lines (or 627 hexadecimal characters) containing all the acquisition information.
The trace measurements values are listed right after the header.

For instance, you could want to save disk space by converting your `.csv` traces files to the binary format and drop the header.
As the metadata will be important to perform the side-channel analysis (plaintext, ciphertext), you must extract them from the filenames.

```python
import re # Regular expressions

def csv2bin(file, offset, type, newfile):
    """ Convert '.csv' file to '.bin' file format 
    
    csv2bin converts a '.csv' input file containing one decimal value per line
    into a '.bin' binary file.
    """
    src_file = open(file, 'r')
    src_file.read(offset)
    lines = src_file.readlines()
    pt_list = []
    
    for line in lines:
        pt_list.append(struct.pack(type, int(line[:-1], base=10)))
    src_file.close()
    
    dest_file = open(newfile, 'wb')
    for point in pt_list:
        dest_file.write(point)
    dest_file.close()

def csv2bin_parallel_call(files, offset, type, reg):
    """ Parallel convert '.csv' file to '.bin' file format."""
    for file in files:
        m = re.search(reg, file)
        if m is None:
            print(file)
        else:
            filename = file[:-3]+"bin"
            Process(target=csv2bin, args=(file, offset,type,filename)).start()
```

```python
# Executing the parallel conversion over all '.csv' files.
regex = '([/a-zA-Z0-9_-]*)(n=[0-9]{1,5}_)([.=a-zA-Z0-9_-]*)'
csv2bin_parallel_call(files, 627, 'h', regex)
```

### Load and visualize traces

```bash
pip install matplotlib
```

```python
import scared
```

For binary files, we need to specify how to build trace set metadata to the traces reader, as a dict:


```python
metadata = {
    'key': scared.traces.bin_extractor.PatternExtractor(r"([A-Fa-f0-9]{32})", num=0),
    'plain': scared.traces.bin_extractor.PatternExtractor(r"([A-Fa-f0-9]{32})", num=1),
    'cipher': scared.traces.bin_extractor.PatternExtractor(r"([A-Fa-f0-9]{32})", num=2)
}
```

Then we create the trace set:

⚠ **Warning:** It can takes **20GB** of RAM.

```python
full_ths = scared.traces.read_ths_from_bin_filenames_pattern(
    './DPA_contest2_public_base_diff_vcc_a128_2009_12_23/*.bin',
    dtype='int16',
    metadatas_parsers=metadata
)
print(full_ths)

# Trace Header Set:
# Name.............: BinFormat trace header set
# Reader...........: Bin format reader with 640000 files, dtype int16
# key..............: uint8
# plain............: uint8
# cipher...........: uint8
```

Since the traces corresponds to 32 subsets of 20,000 traces (One subset per key), we want to perfom the attack on only one subset.

```python
ths = full_ths[0:20000]  # Pick the first subset of traces
```

We can now study the traces. Here, the set contains 20 000 traces, with the same known key for all the traces, and plain and cipher for each trace.


```python
import matplotlib.pyplot as plt
```

Prior to seting up the attack, let's try to focus on the right trace zone. Here, as we have the (plain, cipher) pairs together with the overall encryption measurement, we can try to target the first (yellow) and/or the last AES round (red).


```python
plt.rcParams['figure.figsize']=(21, 5)
data = ths.samples[0:3]
plt.xlabel('Time', fontsize=14)
plt.ylabel('Power', fontsize=14)
plt.plot(data.T)
plt.axvspan(450, 650 , facecolor='y', alpha=0.3, label='First AES round')
plt.axvspan(2340, 2395 , facecolor='r', alpha=0.3, label='Last AES round')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fe7f8195908>




![png](dpa_v2_attack_files/dpa_v2_attack_10_1.png)


---

## Attack the first round with CPA

Here, we will proceed to an attack on the first AES round, with CPA method.
First, we need to define the selection function to be used for intermediate values computation:


```python
import numpy as np
from scared import aes

# The attack selection function takes the targeted meta and guesses as arguments, and returns the intermediate values.
# Here we target the first add round key output of the first aes round.
@scared.attack_selection_function
def first_add_key(plain, guesses):
    res = np.empty((plain.shape[0], len(guesses), plain.shape[1]), dtype='uint8')
    for i, guess in enumerate(guesses):
        res[:, i, :] = np.bitwise_xor(plain, guess)
    return res
```

Second, we can create the CAP attack object, passing it our selection function, a leakage model, and a discriminant function to compute key candidates scores.


```python
cpa_attack = scared.CPAAttack(
    selection_function=first_add_key,
    model=scared.HammingWeight(),
    discriminant=scared.maxabs
)
```

Before running the attack, we finally need to wrap our `TraceHeaderSet` with a trace container, specifiying the analysis frame to use:


```python
first_round_container = scared.Container(ths[:10000], frame=slice(450, 650))
```

We can then proceed to the attack run:


```python
cpa_attack.run(first_round_container)
```

We can check the obtained best key candidate, and compare it to the expected first round:


```python
cpa_first_round_key = np.argmax(cpa_attack.scores, axis=0)
expected_first_round_key = aes.key_schedule(key=ths[0].key)[1]
np.array_equal(expected_first_round_key, cpa_first_round_key)
# False
```




    False



No luck, this attack doesn't retrieve the good key. Let's now try to target the two last rounds of the AES.

## Attack the last rounds with CPA

We first have to define the corresponding selection function for the delta R between last two rounds.


```python
@scared.attack_selection_function
def delta_last_two_rounds(cipher, guesses):
    res = np.empty((cipher.shape[0], len(guesses), cipher.shape[1]), dtype='uint8')
    for i, guess in enumerate(guesses):
        s = aes.inv_sub_bytes(state=np.bitwise_xor(cipher, guess))
        res[:, i, :] = np.bitwise_xor(aes.shift_rows(cipher), s)
    return res
```

We define a new container to target the last two rounds of the trace set:


```python
last_rounds_container = scared.Container(ths[:15000], frame=slice(2340, 2395))
```

We can now create our new CPA analysis. Additionnaly, we use a 500 convergence step, so that we will be able to see how the result depends on the number of traces processed.


```python
cpa_attack = scared.CPAAttack(
    selection_function=delta_last_two_rounds,
    model=scared.HammingWeight(),
    discriminant=scared.maxabs,
    convergence_step=1000
)
```

Let's run the analysis on our container:


```python
cpa_attack.run(last_rounds_container)
```

We can now retrieve the best last round key candidate, and compare it to the expected key:


```python
expected_last_key = aes.key_schedule(key=ths[0].key)[-1]
cpa_last_key = np.argmax(cpa_attack.scores, axis=0)
np.array_equal(expected_last_key, cpa_last_key)
# True
```




    True



Success! The correct last key is retrieved. To finish, we have the ability to study the CPA traces and convergence scores evolution. Let's plot them for the byte 15 of the key:


```python
plt.subplot(121)
plt.title('CPA trace - Byte 15', fontsize=20)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Correlation coefficient value', fontsize=12)
plt.plot(cpa_attack.results[:, 15, :].T)

# Plotting the convergence scores for byte 15
plt.subplot(122)
plt.title('Convergence score - Byte 15', fontsize=20)
plt.xlabel('Number of traces', fontsize=14)
plt.ylabel('CPA score', fontsize=14)
plt.plot(cpa_attack.convergence_traces[:, 15,:].T)
plt.show()
```


![png](dpa_v2_attack_files/dpa_v2_attack_32_0.png)


The best candidate value for this byte is clearly identified, both in correlation traces and convergence traces.
Can we now make the same analysis with DPA method ?

---

## Attack the last round with DPA

We will proceed to the same analysis than before on the last two rounds, but this time with DPA method.
First, we will use a monobit leakage model on the bit 7:


```python
dpa_attack = scared.DPAAttack(
    selection_function=delta_last_two_rounds,
    model=scared.Monobit(7),
    discriminant=scared.maxabs
)
```

We can run the attack:


```python
dpa_attack.run(last_rounds_container)
```

Let's see the obtained key, and compare it to the expected one:


```python
dpa_key = np.argmax(dpa_attack.scores, axis=0)
print(dpa_key)
print(expected_last_key)
# [148 159 123 136 216 184 196  54 116  90  36  95 239 133 192 225]
# [ 83 159 177 136  64 126  43  63  45  90  36  95  80 254 190 225]
```

    [148 159 123 136 216 184 196  54 116  90  36  95 239 133 192 225]
    [ 83 159 177 136  64 126  43  63  45  90  36  95  80 254 190 225]


As we can see, targeting the last bit does not allow us to recover the whole key bytes. Let's try attacking the other bits to retrieve the whole key.


```python
# Let's keep record of the max DPA scores.
max_scores = np.copy(dpa_attack.scores)

for b in range(7):
    print("Bit ", b)
    dpa_analysis = scared.DPAAttack(
        selection_function=delta_last_two_rounds,
        model=scared.Monobit(b),
        discriminant=scared.maxabs
    )
    dpa_analysis.run(last_rounds_container)
    max_scores = np.maximum(max_scores, dpa_analysis.scores)
```

    Bit  0
    Bit  1
    Bit  2
    Bit  3
    Bit  4
    Bit  5
    Bit  6


Now, let's see if we can retrieve all the key bytes:


```python
last_key = np.argmax(max_scores, axis=0)
np.array_equal(last_key, expected_last_key)
# True
```




    True



Success! We have retrieve the whole key.
