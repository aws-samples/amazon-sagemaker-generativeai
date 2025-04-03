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
import pandas as pd
import boto3
import re
import os
from awsUtils import getExtractedDataFromS3


patternIncompleteLine = re.compile(r"([A-Z]*[\.!?]$)", re.M)



class PatternDetector:
    def parseHeaderFooter(self, dataFiltered, occurrence):
        SKIP_PAGES = os.environ['SKIP_PAGES']
        skipPage = SKIP_PAGES.split(",")
        # top 5
        i = 0
        for index, row in dataFiltered.iterrows():
            # skip 2/3 pages
            txt = str(row["Text"])
            # print("header:",txt)
            elemFound = [ele for ele in skipPage if (ele in txt)]
            if elemFound:
                continue

            pair = occurrence.get(txt)
            if pair is None:
                occurrence[txt] = 1
            else:
                pair += 1
                occurrence.update({txt: pair})

            i += 1
            if i > 3:
                break

        dfLast10 = dataFiltered.iloc[-3:]
        for index, row in dfLast10.iterrows():
            txt = str(row["Text"])
            # print("footer:", txt)
            pair = occurrence.get(txt)
            if pair is None:
                occurrence[txt] = 1
            else:
                pair += 1
                occurrence.update({txt: pair})

        return occurrence

    def regexHeaderOrFooter(self, line):
        if len(patternIncompleteLine.findall(line)) == 0:
            length = len(line.split(" "))
            if line[0].isdigit() or line.isupper() or length < 2:
                return True
        return False

    def identifyHeaderFooterPattern(self, bucketName, prefixPath, totalPages):
        headerFooterPattern = []
        i = 1
        page = 1

        occurrence = {}

        try:
            fullScan = True
            while True:
                prefix = f"{prefixPath}/{i}"
                dataFiltered = getExtractedDataFromS3(bucketName, prefix)
                pages = dataFiltered["Page"].unique()
                pages.sort()
                for page in pages:

                    data = dataFiltered[dataFiltered["Page"] == page]
                    # data.sort_values(by=['Top'], ascending=True, inplace=True)

                    totalRow = data.shape[0]
                    if totalRow == 0:
                        break
                    # print("header/footer data for page : ", i , " totalRows :", totalRow)
                    occurrence = self.parseHeaderFooter( data, occurrence )

                    # print("header/footer map ", len(occurrence))
                    #page = 1
                    if (page >= totalPages or page > 20) and fullScan:
                        for k, v in occurrence.items():
                            if v > 10 or v >= totalPages:
                                fullScan = False
                                break
                if not fullScan:
                    break
                i += 1
                # break
        except Exception as e:
            print("end of processing :", i)
            print(e)

        filterdHeaderFooter = {}
        for k, v in occurrence.items():
            if v > 10 or v >= totalPages:
                filterdHeaderFooter[k] = v

        [print(k, ":", v) for k, v in filterdHeaderFooter.items()]
        return filterdHeaderFooter
