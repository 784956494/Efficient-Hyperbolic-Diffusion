n=312
for (( i=1 ; i<=$n ; i++ )); 
do
    mv rec_img$i.png test_autoencoder/mnist/reconstructed
done