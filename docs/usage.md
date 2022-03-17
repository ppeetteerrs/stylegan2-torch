# Usage

## **Important Gotchas**
- Make sure all product dependencies are available on both `conda-forge` and `pypi`
- Repo version tags must be of the format `vX.Y.Z` with lowercase `v`
- If GitHub Actions are not triggering, check [here](https://www.githubstatus.com/) to make sure it is not because of an outrage.
- Upload to Codecov might fail if you commit your repository too fast after creation / if you have not logged in to Codecov via GitHub. Just re-run the GitHub action in that case.


## Things You Can Do
- **Conda Description**: Write a longer and better description for `conda-recipe/meta.yaml`.
- **Extra Branches**: Separate into `dev` or `feature` branches. You might want to add GitHub Action triggers to push / pull requests to those branches.
- **Tests**: Write tests in `pytest`. Other testing framework would require minor changes.
- **Documentation**: Write some nice documentation in the `docs` directory.
- **Improve setup.py**: You can add `description`, `package_data`, `classifiers` and `keywords` to your `setup.py`.