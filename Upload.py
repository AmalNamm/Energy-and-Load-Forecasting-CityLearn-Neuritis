# Add AIcrowd git remote endpoint
git remote add aicrowd git@gitlab.aicrowd.com:phil_aupke/citylearn-2023-starter-kit.git 
git push aicrowd master

# Commit All your changes
git commit -am "Update"

# Create a tag for your submission and push
git tag -am "submission-v0.7" submission-v0.7
git push aicrowd master
git push aicrowd submission-v0.7
