###
 # Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 # SPDX-License-Identifier: MIT-0
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy of this
 # software and associated documentation files (the "Software"), to deal in the Software
 # without restriction, including without limitation the rights to use, copy, modify,
 # merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 # permit persons to whom the Software is furnished to do so.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 # INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 # PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 # HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 # OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 # SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 # Copyright Amazon.com, Inc. and its affiliates. All Rights Reserved.
#   SPDX-License-Identifier: MIT
######
import unittest
import os
import json
from textract_postprocessor import lambda_handler
class TestTextractPdfTextExtractor(unittest.TestCase):

    #os.environ["TEXTRACT_S3_OUTPUT_PATH"] = "s3://simpleasyncworkflow-textractsimpleasyncworkflow2d-113764vgh21fm/textract-output/UNDP-GEF Terminal Evaluation of wind renewable energy technology project2022-09-15T14:58:13.674106/UNDP-GEF Terminal Evaluation of wind renewable energy technology project.json"
    os.environ['SKIP_PAGES'] = "CONTENTS, TABLE OF CONTENTS, FOREWORDS, ANNEXES,Table of Contents,ACRONYMS,ABBREVIATIONS"
    os.environ['NO_LINES_HEADER'] = "3"
    os.environ['NO_LINES_FOOTER'] = "3"
    os.environ['FILTER_PARA_WORDS'] = "10"
    def test_pdfTextExtractor(self):
        event = "{\"numberOfPages\":2, \"textract_result\":{\"TextractTempOutputJsonPath\": \"s3://simpleasyncworkflow-textractsimpleasyncworkflow2d-gpnimual3cpr/textract-temp-output/f53300f9c94cb1ac5a0d52cec46327038781a264dfc0dc9b87c1f1375a0db368\"}}"
        #event = "{\"numberOfPages\":4, \"textract_result\":{\"TextractTempOutputJsonPath\": \"s3://simpleasyncworkflow-textractsimpleasyncworkflow2d-gpnimual3cpr/textract-temp-output/3c7b88b8327573e87712bb83244a7cd6835f07536110e84df5d9dec89cf0df69\"}}"
        lambda_handler( json.loads(event), "")




if __name__ == '__main__':
    unittest.main()
