"""Run curriculum pre-evaluation, 1:1 fine-tuning, and post-evaluation."""

import sys

from finetuning.run import main


if __name__ == "__main__":
    if "--evaluate-pretrained" not in sys.argv:
        sys.argv.append("--evaluate-pretrained")
    if "--post-label" not in sys.argv:
        sys.argv.extend(("--post-label", "post_1x"))
    main()
