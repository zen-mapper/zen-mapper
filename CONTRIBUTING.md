# Contributing

Thank you for your interest in zen-mapper! Here are a couple of notes on how to
contribute.

## Something seems wrong

If something seems wrong please do not hesitate to open up an
[issue](/issues/new) here on github. Issues are also the best way to ask
questions about zen-mapper. All we ask is before doing so check to make sure
that your questions is not covered by our incredibly sparse
[documentation](https://zen-mapper.github.io/zen-mapper/).

## Contributing code

The general pipeline for getting code added to zen-mapper is as follows:

1. [fork zen-mapper](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)
2. [create a branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches)
3. write your code
4. [open a pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)

Before submitting a new feature it is wise to open an issue discussing the
proposed feature to make sure that there is interest first. Beyond that we can
hash everything out during the review process.

### Developer Environment

If you already use [nix](https://nixos.org) the easiest way to get up and
running is to run `nix develop` which will drop you into a shell with
everything you need. We use [just](https://just.systems) as our task runner,
running `just` will list all the tasks you can run.

If you don't use nix, we provide files for using uv to manage your environment.
Instructions on how to do that can be found
[here](https://docs.astral.sh/uv/guides/projects/).
