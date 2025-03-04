import os
import json
from collections import defaultdict


code_example = """{
  "model_name": "example",
  "model_type": "AI21-SUMMARY",
  "endpoint_name": "summarize",
  "payload": {
    "parameters": {
      "max_length": {
        "default": 200,
        "range": [
          10,
          500
        ]
      },
      "num_return_sequences": {
        "default": 10,
        "range": [
          0,
          10
        ]
      },
      "num_beams": {
        "default": 3,
        "range": [
          0,
          10
        ]
      },
      "temperature": {
        "default": 0.5,
        "range": [
          0,
          1
        ]
      },
      "early_stopping": {
        "default": true,
        "range": [
          true,
          false
        ]
      },
      "stopwords_list": {
        "default": [
          "stop",
          "dot"
        ],
        "range": [
          "a",
          "an",
          "the",
          "and",
          "it",
          "for",
          "or",
          "but",
          "in",
          "my",
          "your",
          "our",
          "stop",
          "dot"
        ]
      }
    }
  }
}
"""

parameters_help_map = {
    "max_length": "Model generates text until the output length (which includes the input context length) reaches max_length. If specified, it must be a positive integer.",
    "num_return_sequences": "Number of output sequences returned. If specified, it must be a positive integer.",
    "num_beams": "Number of beams used in the greedy search. If specified, it must be integer greater than or equal to num_return_sequences.",
    "no_repeat_ngram_size": "Model ensures that a sequence of words of no_repeat_ngram_size is not repeated in the output sequence. If specified, it must be a positive integer greater than 1.",
    "temperature": "Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If temperature -> 0, it results in greedy decoding. If specified, it must be a positive float.",
    "early_stopping": "If True, text generation is finished when all beam hypotheses reach the end of stence token. If specified, it must be boolean.",
    "do_sample": "If True, sample the next word as per the likelyhood. If specified, it must be boolean.",
    "top_k": "In each step of text generation, sample from only the top_k most likely words. If specified, it must be a positive integer.",
    "top_p": "In each step of text generation, sample from the smallest possible set of words with cumulative probability top_p. If specified, it must be a float between 0 and 1.",
    "seed": "Fix the randomized state for reproducibility. If specified, it must be an integer.",
}

example_list = [" ", "Table Q&A", "Product description", "Summarize reviews", "Generate SQL"]
example_context_ai21_qa = ["Sample Context ", "Financial", "Healthcare"]
example_prompts_ai21 = {
    "Table Q&A": "| Ship Name | Color | Total Passengers | Status | Captain | \n \
| Symphony | White | 7700 | Active | Mike | \n \
| Wonder | Grey | 7900 | Under Construction | Anna | \n \
| Odyssey | White | 5800 | Active | Mohammed | \n \
| Quantum | White | 5700 | Active | Ricardo | \n \
| Mariner | Grey | 4300 | Active | Saanvi | \n \
Q: Which active ship carries the most passengers? \n \
A: Symphony \n \
Q: What is the color of the ship whose captain is Saanvi? \n \
A: Grey \n \
Q: How many passengers does Ricardo's ship carry? \n \
A:",
    "Product description": "Write an engaging product description for clothing eCommerce site. Make sure to include the following features in the description. \n \
Product: Humor Men's Graphic T-Shirt with a print of Einstein's quote: \"artificial intelligence is no match for natural stupidity” \n \
Features: \n \
- Soft cotton \n \
- Short sleeve \n \
Description:",
    "Summarize reviews": "Summarize the following restaurant review \n \
Restaurant: Luigi's \n \
Review: We were passing through SF on a Thursday afternoon and wanted some Italian food. We passed by a couple places which were packed until finally stopping at Luigi's, mainly because it was a little less crowded and the people seemed to be mostly locals. We ordered the tagliatelle and mozzarella caprese. The tagliatelle were a work of art - the pasta was just right and the tomato sauce with fresh basil was perfect. The caprese was OK but nothing out of the ordinary. Service was slow at first but overall it was fine. Other than that - Luigi's great experience! \n \
Summary: Local spot. Not crowded. Excellent tagliatelle with tomato sauce. Service slow at first. \n \
## \n \
Summarize the following restaurant review \n \
Restaurant: La Taqueria \n \
Review: La Taqueria is a tiny place with 3 long tables inside and 2 small tables outside. The inside is cramped, but the outside is pleasant. Unfortunately, we had to sit inside as all the outside tables were taken. The tacos are delicious and reasonably priced and the salsa is spicy and flavorful. Service was friendly. Aside from the seating, the only thing I didn't like was the lack of parking - we had to walk six blocks to find a spot. \n \
Summary:",
    
"Generate SQL": "Create SQL statement from instruction. \n \
Database: Customers(CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country)\n \
Request: all the countries we have customers in without repetitions.\n \
SQL statement:\n \
SELECT DISTINCT Country FROM Customers;\n \
##\n \
Create SQL statement from instruction.\n \
Database: Orders(OrderID, CustomerID, EmployeeID, OrderDate, ShipperID)\n \
Request: select all the orders from customer id 1.\n \
SQL statement:\n \
SELECT * FROM Orders\n \
WHERE CustomerID = 1;\n \
##\n \
Create SQL statement from instruction.\n \
Database: Products(ProductID, ProductName, SupplierID, CategoryID, Unit, Price)\n \
Request: selects all products from categories 1 and 7\n \
SQL statement:\n \
SELECT * FROM Products\n \
WHERE CategoryID = 1 OR CategoryID = 7;\n \
##\n \
Create SQL statement from instruction.\n \
Database: Customers(CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country)\n \
Request: change the first customer's name to Alfred Schmidt who lives in Frankfurt city.\n \
SQL statement:",
}

example_context_ai21_qa = {
    "Financial": "n 2020 and 2021, enormous QE — approximately $4.4 trillion, or 18%, of 2021 gross domestic product (GDP) — and enormous fiscal stimulus (which has been and always will be inflationary) — approximately $5 trillion, or 21%, of 2021 GDP — stabilized markets and allowed companies to raise enormous amounts of capital. In addition, this infusion of capital saved many small businesses and put more than $2.5 trillion in the hands of consumers and almost $1 trillion into state and local coffers. These actions led to a rapid decline in unemployment, dropping from 15% to under 4% in 20 months — the magnitude and speed of which were both unprecedented. Additionally, the economy grew 7% in 2021 despite the arrival of the Delta and Omicron variants and the global supply chain shortages, which were largely fueled by the dramatic upswing in consumer spending and the shift in that spend from services to goods. Fortunately, during these two years, vaccines for COVID-19 were also rapidly developed and distributed. \n \
In today's economy, the consumer is in excellent financial shape (on average), with leverage among the lowest on record, excellent mortgage underwriting (even though we've had home price appreciation), plentiful jobs with wage increases and more than $2 trillion in excess savings, mostly due to government stimulus. Most consumers and companies (and states) are still flush with the money generated in 2020 and 2021, with consumer spending over the last several months 12% above pre-COVID-19 levels. (But we must recognize that the account balances in lower-income households, smaller to begin with, are going down faster and that income for those households is not keeping pace with rising inflation.) \n \
Today's economic landscape is completely different from the 2008 financial crisis when the consumer was extraordinarily overleveraged, as was the financial system as a whole — from banks and investment banks to shadow banks, hedge funds, private equity, Fannie Mae and many other entities. In addition, home price appreciation, fed by bad underwriting and leverage in the mortgage system, led to excessive speculation, which was missed by virtually everyone — eventually leading to nearly $1 trillion in actual losses.",
    "Healthcare": "A heart attack occurs when blood flow that brings \
oxygen-rich blood to the heart muscle is severely \
reduced or cut off. This is due to a buildup of fat,\
cholesterol and other substances (plaque) that narrows\
coronary arteries. This process is called atherosclerosis.\
When plaque in a heart artery breaks open, a blood clot\
forms. The clot can block blood flow. When it completely\
stops blood flow to part of the heart muscle, that\
portion of muscle begins to die. Damage increases the\
longer an artery stays blocked. Once some of the heart\
muscle dies, permanent heart damage results.\
The amount of damage to the heart muscle depends on\
the size of the area supplied by the blocked artery and\
the time between injury and treatment. The blocked\
artery should be opened as soon as possible to reduce\
heart damage. \n \
Atherosclerosis develops over time. It often has no symptoms\
until enough damage lessens blood flow to your heart\
muscle. That means you usually can’t feel it happening until\
blood flow to heart muscle is blocked. \n \
You should know the warning signs of heart attack so you\
can get help right away for yourself or someone else.\
Some heart attacks are sudden and intense. But most start\
slowly, with mild pain or discomfort. Signs of a heart attack\
include:\n\
• Uncomfortable pressure, squeezing, fullness or pain in the\
center of your chest. It lasts more than a few minutes, or\
goes away and comes back.\n\
• Pain or discomfort in one or both arms, your back, neck,\
jaw or stomach.\n\
• Shortness of breath with or without chest discomfort.\n\
• Other signs such as breaking out in a cold sweat, nausea\
or lightheadedness."
}
parameters_help_map = defaultdict(str, parameters_help_map)
example_prompts_ai21 = defaultdict(str, example_prompts_ai21)
example_context_ai21_qa = defaultdict(str, example_context_ai21_qa)
