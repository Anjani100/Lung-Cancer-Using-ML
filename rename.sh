find ./ -type f -follow -print > files.txt 
grep '.ipynb' files.txt > files1.txt
grep -v '.ipynb_checkpoints' files1.txt > files2.txt

# jupyter nbconvert 'Untitled.ipynb' --to python

input="./files2.txt"

while IFS= read -r line
do
  echo "$line"
  jupyter nbconvert "$line" --to python
done < "$input"

rm files.txt
rm files1.txt
rm files2.txt