pip install -r requirements.txt
cp ../*.md source/
sphinx-build -M clean source build
sphinx-build -M html source build
mv build/html ../public/
