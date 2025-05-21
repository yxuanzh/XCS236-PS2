rm -f ps2.zip 
pushd submission; zip -r ../ps2.zip . --exclude "__pycache__/*"; popd