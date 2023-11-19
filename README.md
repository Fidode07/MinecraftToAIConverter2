# 🤖 MinecraftToAIConverter 🤖

MinecraftToAIConverter<strong>2</strong> is the successor of my old
project <a href="https://github.com/Fidode07/MinecrafToAIConverter">MinecraftToAIConverter</a>. Since I didn't know much
about artificial intelligence
and Python back then, it was more of a "project to learn".
Therefore, after almost 2 years, I finally decided to revise the project. Well, actually I rather rewrote it.

# 🛠️ New features &amp; optimized things 🛠️

The old version had many problems. A lot, in fact. I have improved that:

- <strong>Structured Code</strong>: The old version was a bunch of files and scripts thrown together. I didn't even
  really use classes. In the new version, everything is in a functional structure.
- <strong>Translation to TensorFlow</strong>: I don't want to start a dispute or a discussion, so let's see this neither
  as a contra nor as a pro. I have translated the model from PyTorch to TensorFlow.
- <strong>Word2Vec</strong>: The old function worked using the well-known Bag of Words method. What we are all thinking
  now: Shit. But don't worry, this has of course been revised. The new version uses a simple and easily customizable
  Word2Vec class.
- <strong>Multiple datasets</strong>: Yes, you heard right. The Train function automatically merges your datasets as
  long as they all follow the structure described below. So you can swap out all your data and don't have to keep a 1GB
  .json file.
- <strong>Usable response</strong>: The old Python server actually only had a random response that belonged to the tag.
  So the plugin behind it couldn't do anything with it except send it directly to the player. The new code responds in
  the JSON response, which makes it possible to have more control in the plugin itself. For example, you can add custom
  quests and menus.

# ❓ How do I integrate this into my plugins? ❓

You can integrate the whole thing into your own plugins however you like. If you want the AI to answer something, simply
open a TCP socket to the Python backend. But please do not forget that you have to adapt the data set beforehand and then re-execute the Python script.
You must send the request in json format, and it must look like this:

```json
{
  "sentence": "<what-the-user-wrote>"
}
```

The Python backend will then return two possible answers. If everything was successful:

```json
{
  "status": "ok",
  "sentence": "<what-the-user-wrote>",
  "tag": "<classified-tag>",
  "responses": "<list-of-responses>",
  "confidence": "<confidence>"
}
```

- <strong>status</strong>: The status actually only has 2 options. 'error' or 'ok'. With an error, something has gone
  wrong, with an ok, everything is fine
- <strong>sentence</strong>: This is simply the sentence that was sent to the AI
- <strong>tag</strong>: The tag contains the classified tag. As you can see in the dataset structure below, all example
  records have a large tag. This allows you to tell what the user actually meant
- <strong>responses</strong>: Responses is a list of strings. To be precise, it contains the responses specified in the
  dataset
- <strong>confidence</strong>: Confidence tells you how safe the model is. How likely she thinks it is that the day is
  really the right one.

<strong>Or if an error has occurred:</strong>

```json
{
  "status": "error",
  "error_msg": "<some-error-msg>"
}
```

- <strong>status</strong>: Same as above
- <strong>error_msg</strong>: An error message that tells you exactly what was wrong

# 🔌 Sounds difficult, any examples? 🔌

Sure, I have written an example plugin. Go to the
following <a href="https://github.com/Fidode07/MinecraftToAIConverter-ExamplePlugin">repo</a>. What the plugin does in
short:

You can talk to a villager close to you with the command ``/ask <msg>``. Your sentence is sent to an AI, which
classifies it.

# 📊 Dataset structure 📊

The Dataset is pretty basic. The code is easily extendable, so if you are missing something, have fun extending it.
Anyway, here are all the parameters that needs to be given:

- <strong>tag</strong>: A unique tag/identifier for the intent
- <strong>patterns</strong>: A list of strings that contains all your example sentences
- <strong>responses</strong>: A list of strings containing all your answer sentences

<strong>NOTE: Your intents must be in an "intents" list, which is absolute.</strong>
Example:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "hello",
        "hi",
        "hey",
        "howdy",
        "good morning",
        "good afternoon",
        "greetings",
        "whats up",
        "hows it going"
      ],
      "responses": [
        "hello",
        "hi there",
        "hey",
        "howdy",
        "good day how can i assist you",
        "greetings what brings you here",
        "hello how can i help you"
      ]
    },
    {
      "tag": "bye",
      "patterns": [
        "adios",
        "bye",
        "see you",
        "see ya",
        "farewell",
        "goodbye",
        "take care",
        "until next time"
      ],
      "responses": [
        "goodbye",
        "farewell",
        "see you later",
        "take care",
        "until next time",
        "adios",
        "bye for now",
        "see ya"
      ]
    }
  ]
}
```
