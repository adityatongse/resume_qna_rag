# CV Directory

Place your CV file here.

## Supported Formats

- PDF (`.pdf`)
- DOCX (`.docx`)

## Instructions

1. Copy your CV to this directory
2. Update the `CV_FILE_PATH` in your `.env` file to point to your CV

Example:
```bash
cp ~/Documents/my_resume.pdf data/cv/resume.pdf
```

Then in `.env`:
```
CV_FILE_PATH=data/cv/resume.pdf
```

## Note

Your CV file is gitignored for privacy. The actual CV files will not be committed to version control.