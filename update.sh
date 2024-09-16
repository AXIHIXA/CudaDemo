if [ -z ${1+x} ]
then
    git add -A &&
    git commit -m "Update" &&
    git push
else
    git add -A &&
    git commit -m "\"$1\"" &&
    git push
fi



