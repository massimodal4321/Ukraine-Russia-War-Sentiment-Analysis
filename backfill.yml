name: Backfill

# Chains 6-hour jobs so it keeps working through the night.
# Each run picks up where the last one stopped.
on:
  workflow_dispatch:
    inputs:
      phase:
        description: "Sampling density"
        required: true
        default: monthly
        type: choice
        options: [monthly, 10daily, weekly, every3rd, daily]
      chain:
        description: "Re-trigger itself when the budget runs out"
        required: true
        default: "true"
        type: choice
        options: ["true", "false"]

permissions:
  contents: write
  actions: write

concurrency:
  group: backfill
  cancel-in-progress: false

jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 350
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install
        run: pip install -q google-genai google-cloud-bigquery db-dtypes py3langid

      - name: Write service account key
        run: echo '${{ secrets.GCP_SA_KEY }}' > /tmp/sa.json

      - name: Backfill
        env:
          GEMINI_KEY: ${{ secrets.GEMINI_KEY }}
          GCP_SA_KEY_FILE: /tmp/sa.json
          PHASE: ${{ inputs.phase }}
          BUDGET_MIN: "320"
        run: python scripts/backfill.py

      - name: Commit
        run: |
          git config user.name  "backfill-bot"
          git config user.email "actions@github.com"
          git add data/daily.json
          if git diff --staged --quiet; then
            echo "nothing new"
          else
            git commit -m "backfill: ${{ inputs.phase }} $(date -u +%Y-%m-%dT%H:%MZ)"
            git pull --rebase --autostash
            git push
          fi

      - name: Chain another run if work remains
        if: ${{ inputs.chain == 'true' }}
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          REMAIN=$(cat data/.remaining 2>/dev/null || echo 0)
          echo "days remaining: $REMAIN"
          if [ "$REMAIN" -gt 0 ]; then
            gh workflow run backfill.yml -f phase=${{ inputs.phase }} -f chain=true
            echo "chained another run"
          else
            echo "phase complete, stopping"
          fi
