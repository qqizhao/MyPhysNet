conda remove --name rppg --all -y
conda create -n rppg python=3.8 pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=10.2 -c pytorch -q -y
