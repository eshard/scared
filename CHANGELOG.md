# Changelog

### <small>0.9.2 (2022-09-16)</small>
* Resolve "Bug template attack selection function verification"
* Resolve "Update CI image"
* Resolve "optimization Second order preprocess with distance"
* Replace np.min/max with python min/maxab
* Fix lint

### <small>0.9.1 (2022-07-11)</small>
* Changed mail support
* Changed CI and version build

### <small>0.8.13 (2020-10-14)</small>

* maint: Add tests for python 3.8. ([22b0098e](https://gitlab.com/eshard/scared/-/commit/22b0098e90f8b875faee93c32b84b1cdd31a3934))
* fix: Update example in redme. ([b22c49db](https://gitlab.com/eshard/scared/-/commit/b22c49db5cefba319ea32cefb1ea8f7be5670fa8))

### <small>0.8.12 (2020-06-16)</small>

* fix: fix bug in template distinguisher ([f62f2d7](https://gitlab.com/eshard/scared/commit/f62f2d7))

### <small>0.8.11 (2019-12-17)</small>

* fix: compute batch size private method working statically ([2cc26d8](https://gitlab.com/eshard/scared/commit/2cc26d8))

### <small>0.8.10 (2019-12-16)</small>

* fix: put back private compute batch size method on container ([eeb7e12](https://gitlab.com/eshard/scared/commit/eeb7e12))

### <small>0.8.9 (2019-12-13)</small>

* fix: avoid iterable conversion to list ([0b9957d](https://gitlab.com/eshard/scared/commit/0b9957d))

### <small>0.8.8 (2019-12-12)</small>

* fix: avoid loading too much traces for trace_size computation ([88fd69c](https://gitlab.com/eshard/scared/commit/88fd69c))

### <small>0.8.7 (2019-12-12)</small>

* fix: container trace size and batch size depend on preprocesses ([8041375](https://gitlab.com/eshard/scared/commit/8041375))
* maint: add instructions for python 3.8 support ([09c3609](https://gitlab.com/eshard/scared/commit/09c3609))
* maint: update CHANGELOG ([b93442c](https://gitlab.com/eshard/scared/commit/b93442c))
* doc: improve README ([6b44dc5](https://gitlab.com/eshard/scared/commit/6b44dc5))

### <small>0.8.6-doc</small> (2019-12-09)

* maint: add instructions for python 3.8 support ([09c3609](https://gitlab.com/eshard/scared/commit/09c3609))
* doc: improve README ([6b44dc5](https://gitlab.com/eshard/scared/commit/6b44dc5))

### <small>0.8.6 (2019-11-25)</small>

* fix: documentation, pypi source build and conda build now working for C extension ([29dfb27](https://gitlab.com/eshard/scared/commit/29dfb27))

### <small>0.8.5 (2019-11-14)</small>

* fix: backward and forward AES key expansion works starting from any round ([a37072b](https://gitlab.com/eshard/scared/commit/a37072b))

### <small>0.8.4 (2019-10-04)</small>

* fix: optimization of AES and DES ciphers performance ([c080a56](https://gitlab.com/eshard/scared/commit/c080a56))

### <small>0.8.3 (2019-10-04)</small>

* fix: improve string representation performance of Container object ([7ce760f](https://gitlab.com/eshard/scared/commit/7ce760f))

### 0.8.0 (2019-09-25)

* feat: New signal processing functions + Synchronizer API added ([2272602](https://gitlab.com/eshard/scared/commit/2272602))

### <small>0.7.1 (2019-09-06)</small>

* fix: container string representation now working with all kind of preprocesses ([38dd64d](https://gitlab.com/eshard/scared/commit/38dd64d))

### 0.7.0 (2019-08-30)

* fix: remove unused private APIs ([8fe310f](https://gitlab.com/eshard/scared/commit/8fe310f))
* feat:HammingWeight computation optimization, about 15 time faster, and accept all unsigned integer d ([771d221](https://gitlab.com/eshard/scared/commit/771d221))

### <small>0.6.2 (2019-08-26)</small>

* fix: Update AES add round key selection functions to be work with non contiguous guesses ([246473b](https://gitlab.com/eshard/scared/commit/246473b))
* fix: Update DES add round key function to work with non contiguous guesses ([022feb1](https://gitlab.com/eshard/scared/commit/022feb1))
* doc: Improve API doc and DPA V2 guide documentation. ([f93de07](https://gitlab.com/eshard/scared/commit/f93de07))

### <small>0.6.1 (2019-08-22)</small>

* maint: Conda use noarch build ([08e21fb](https://gitlab.com/eshard/scared/commit/08e21fb))
* maint: Prepare 0.6.1 release ([1f1630f](https://gitlab.com/eshard/scared/commit/1f1630f))

### 0.6.0 (2019-08-13)

* maint: Prepare 0.6.0 release ([3b3ac75](https://gitlab.com/eshard/scared/commit/3b3ac75))
* feat: Add DES APIs and selection functions ([8192752](https://gitlab.com/eshard/scared/commit/8192752))

### 0.5.0 (2019-07-26)

* maint: Prepare 0.5.0 release ([7f40d02](https://gitlab.com/eshard/scared/commit/7f40d02))
* maint: Update conda build version ([9bd667f](https://gitlab.com/eshard/scared/commit/9bd667f))
* feat: Add template attacks. ([7488236](https://gitlab.com/eshard/scared/commit/7488236))

### <small>0.4.1 (2019-07-23)</small>

* maint: Adjust some log messages in analysis process. ([5630781](https://gitlab.com/eshard/scared/commit/5630781))
* maint: prepare 0.4.1a release ([08fac74](https://gitlab.com/eshard/scared/commit/08fac74))
* doc: add pretty strings to APIs objects ([92ab135](https://gitlab.com/eshard/scared/commit/92ab135))
* fix: CenterOn preprocess handle None mean gracefully ([50b72de](https://gitlab.com/eshard/scared/commit/50b72de))
* fix: High order combination preprocesses accepts frame_1 default value. ([2c68a32](https://gitlab.com/eshard/scared/commit/2c68a32))
* feat: add expected key function to AES ready-to-use selection functions. ([429ddd1](https://gitlab.com/eshard/scared/commit/429ddd1))
* feat: Add expected_key_function option to attack selection function decorator. ([7351d0b](https://gitlab.com/eshard/scared/commit/7351d0b))
* feat: add memory usage estimation check at first distinguisher update ([0a86768](https://gitlab.com/eshard/scared/commit/0a86768))
* feat: add private _Analysis API to create analysis objects from standalone distinguishers instances. ([5123d83](https://gitlab.com/eshard/scared/commit/5123d83))
* feat: Add Reverse and Attacks specialized classes for each type of distinguisher analysis. ([76b80bb](https://gitlab.com/eshard/scared/commit/76b80bb))
* feat: BREAKING CHANGE - AES selections functions are under aes package and are now classes. ([68c5a2f](https://gitlab.com/eshard/scared/commit/68c5a2f))
* feat: High order combination preprocesses raise exception if improper mode is passed. ([3e0a76d](https://gitlab.com/eshard/scared/commit/3e0a76d))
* feat: Improve selection function error messages ([4724ba1](https://gitlab.com/eshard/scared/commit/4724ba1))
* feat: rename t-Test container arguments to ths_1 and ths_2 ([8e9e3e1](https://gitlab.com/eshard/scared/commit/8e9e3e1))
* feat: selection_function accepts range guesses and list words. ([7fe1014](https://gitlab.com/eshard/scared/commit/7fe1014))

### 0.3.0 (2019-07-09)

* maint: Prepare 0.3.0 release ([4010dd4](https://gitlab.com/eshard/scared/commit/4010dd4))
* feat: Add MIA distinguisher and analysis APIs. ([a65fffe](https://gitlab.com/eshard/scared/commit/a65fffe))
* feat: Add t-test analysis API ([a62b054](https://gitlab.com/eshard/scared/commit/a62b054))
* feat: Added traces partitioning distinguishers ANOVA, NICV ans SNR. ([177e33b](https://gitlab.com/eshard/scared/commit/177e33b))

### 0.2.0 (2019-06-26)

* packaging: Prepare 0.2.0 version tagging ([63a3a90](https://gitlab.com/eshard/scared/commit/63a3a90))
* feat: Add AES ready-to-use attack selection functions. ([1859977](https://gitlab.com/eshard/scared/commit/1859977))
* feat: Add basic preprocesses - square, fft modulus, serialize bits, center and standardization. ([80211b6](https://gitlab.com/eshard/scared/commit/80211b6))
* feat: Add preprocess decorator and preprocesses to container ([a3db2ad](https://gitlab.com/eshard/scared/commit/a3db2ad))
* feat: add preprocesses for high order analysis, both standard and time/frequency combinations. ([679c822](https://gitlab.com/eshard/scared/commit/679c822))
* doc: add Gitlab issue templates ([fe092db](https://gitlab.com/eshard/scared/commit/fe092db))
* doc: Update CONTRIBUTING guide with merge request checklist ([ab123cc](https://gitlab.com/eshard/scared/commit/ab123cc))

### <small>0.1.1 (2019-06-14)</small>

* fix: slight updates to documentation and README ([a7bee79](https://gitlab.com/eshard/scared/commit/a7bee79))

### 0.1.0 (2019-06-14)

* feat: initialize library public repository for 0.1.0 alpha release ([6c2a41d](https://gitlab.com/eshard/scared/commit/6c2a41d))
