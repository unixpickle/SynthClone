mkdir books

for book in {0..100}; do
  curl https://www.gutenberg.org/cache/epub/$book/pg${book}.txt >books/${book}.txt
done
