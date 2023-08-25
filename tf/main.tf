terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region  = "us-east-1"
}

resource "aws_s3_bucket" "nail_numera_data" {
  bucket = "nail-numerai-data"

  tags = {
    Project = "nail"
  }
}

resource "aws_s3_bucket_policy" "lightning_ai_access_nail_numerai_data" {
  bucket = aws_s3_bucket.nail_numera_data.id
  policy = data.aws_iam_policy_document.lightning_ai_access_nail_numerai_data.json
}

data "aws_iam_policy_document" "lightning_ai_access_nail_numerai_data" {
  statement {
    sid = "GrantLightningAIAccess"

    principals {
      type        = "AWS"
      identifiers = ["748115360335"]
    }

    actions = [
      "s3:*",
    ]

    resources = [
      aws_s3_bucket.nail_numera_data.arn,
      "${aws_s3_bucket.nail_numera_data.arn}/*",
    ]

    condition {
      test     = "ForAnyValue:StringEquals"
      variable = "aws:PrincipalTag/lightning/project"

      values = [
        "01h7bjhf9445v10zm7m2s9hk6h"
      ]
    }
  }
}
