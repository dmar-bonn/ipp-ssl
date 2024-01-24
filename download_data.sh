curl -LO https://phenoroam.phenorob.de/file-uploader/download/public/6314864478-worlds.zip
curl -LO https://phenoroam.phenorob.de/file-uploader/download/public/4930684482-potsdam.zip
curl -LO https://phenoroam.phenorob.de/file-uploader/download/public/244704144-pretrained.zip

unzip 6314864478-worlds.zip -d .
unzip 4930684482-potsdam.zip -d bayesian_erfnet/agri_semantics/datasets/
unzip 244704144-pretrained.zip -d bayesian_erfnet/agri_semantics/models/

rm 6314864478-worlds.zip 4930684482-potsdam.zip 244704144-pretrained.zip
