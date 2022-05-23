# pyklab
This repository is a library for Materials Informatics published by Assistant Professor <a href="https://researchmap.jp/mkumagai?lang=en">Masaya Kumagai</a> at Kyoto University.

## ENVIRONMENT SETUP
Set up an environment with a directory tree structure like the following.
```
|-YOUR_WORKSPACE
  |- pyklab(<-get it from Github)
    |- pyklab
    |- workdir
      |- ~~.ipynb
```

### <b>○ Docker-compose</b>

> <b>Requirements</b></br>
> * Git (windows user): https://gitforwindows.org/
> * Docker Desktop: https://matsuand.github.io/docs.docker.jp.onthefly/desktop/

Run the following commands in a terminal (command prompt).

```sh
cd YOUR_WORKSPACE
git clone https://github.com/kumagallium/pyklab.git
cd pyklab
mkdir workdir
docker-compose run up
```
※Please tune docker desktop resources as needed.

## How to use
1.  Put your program code in the "workdir" directory
2.  Import this pyklab library
```python
%load_ext autoreload
%autoreload 2
import sys
sys.path.append('../')
from pyklab import pyklab
```

## Contributing
1. Fork it (`git clone https://github.com/kumagallium/pyklab.git`)
2. Create your feature branch (`git checkout -b your-new-feature`)
3. Commit your changes (`git commit -am 'feat: add some feature'`)
4. Push to the branch (`git push origin your-new-feature`)
5. Create a new Pull Request

## License
pyklab is developed and maintained by Masaya Kumagai, under [MIT License](LICENSE).