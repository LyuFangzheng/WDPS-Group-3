from pyspark.context import SparkContext
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import nltk
import re
import os
from io import BytesIO
from warcio.archiveiterator import ArchiveIterator

from bs4 import BeautifulSoup

# nltk.download() # Use only if not yet installed


def ner_stanford(input, st):

    output = []

    for i in range(len(input)):
        tokenized_text = word_tokenize(input[1])
        classified_text = st.tag(tokenized_text)

        for tup in classified_text:
            if tup[1] != 'O':
                output.append(tup[0] + ',' + tup[1] + ',' + input[0])


    output = list(set(output))
    return output


# defines which tags are excluded from the HTML file
def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', element):
        return False
    return True


def decode(x, record_attribute):
    html_pages_array = []

    _, payload = x

    wholeTextFile = ''.join([c for c in payload])
    wholeTextFile = "WARC/1.0 " + wholeTextFile
    wholeTextFile = wholeTextFile.encode('utf-8')
    # print(wholeTextFile)



    stream = BytesIO(wholeTextFile)

    list_error = [] # This is to see where the error occured
    try:
        for record in ArchiveIterator(stream):

            # if the record type is a response (which is the case for html page)
            list_error.append('1')
            if record.rec_type == 'response':
                list_error.append('2')
                # check if the response is http
                if record.http_headers != None:
                    list_error.append('3')
                    # Get the WARC-RECORD-ID
                    record_id = record.rec_headers.get_header(record_attribute)
                    list_error.append('4')
                    # Clean up the HTML using BeautifulSoup
                    html = record.content_stream().read()
                    soup = BeautifulSoup(html, "html5lib")
                    data = soup.findAll(text=True)#.encode()
                    list_error.append('5')
                    result = filter(visible, data)
                    list_error.append('5.1')
                    result2 = ' '.join(result)
                    list_error.append('5.2')
                    result2 = ' '.join(result2.split())
                    list_error.append('6')
                    # Build up the resulting list.
                    list_error.append('7')
                    result2 = result2.encode('ascii', errors="ignore").decode('ascii')
                    list_error.append('7.1')
                    if result2 != '' and isinstance(result2, str):
                        html_pages_array.append([record_id, result2])
                        list_error.append('8')

    except Exception:
        print("Something went wrong with the archive entry")
        print(list_error)


    return html_pages_array



java_path = "C:/Program Files/Java/jdk1.8.0_191/bin/java.exe"
os.environ['JAVAHOME'] = java_path

record_attribute = "WARC-Record-ID"
# Here we use a smaller testfile due to computation time. Use the sample.war.gz for real testing.
in_file = "C:/Users/klm85310/Documents/WDPS/testing.warc.gz"
stanford = 'C:/Users/klm85310/Documents/WDPS/stanford-ner-2017-06-09/stanford-ner-2017-06-09'

# Create Spark Context -- Remove this when running on cluster
sc = SparkContext.getOrCreate()
st = StanfordNERTagger(stanford + '/classifiers/english.all.3class.distsim.crf.ser.gz',
                       stanford + '/stanford-ner.jar',
                       encoding='utf-8')

rdd_whole_warc_file = rdd = sc.newAPIHadoopFile(in_file,
                                                "org.apache.hadoop.mapreduce.lib.input.TextInputFormat",
                                                "org.apache.hadoop.io.LongWritable",
                                                "org.apache.hadoop.io.Text",
                                                conf={"textinputformat.record.delimiter": "WARC/1.0"})

rdd_html_cleaned = rdd_whole_warc_file.flatMap(lambda x: decode(x, record_attribute))

print("step 2")


# Extract named Entities
stanford_rdd = rdd_html_cleaned.map(lambda x: ner_stanford(x, st))


#print(stanford_rdd.collect())
stanford_rdd.saveAsTextFile('Entities_short.csv')
print('Done')