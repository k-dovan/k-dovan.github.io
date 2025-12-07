---
title: "From Model to Production: Deploying Deep Learning Without the Drama"
discription: "Creating a model is the rehearsal; deploying it is the live concert — with no room for off-key notes."
date: 2025-09-30 09:30:00 +0700
categories: [Deep learning for coders with fastai and PyTorch]
tags: [drivetrain approach,data augmentation,data cleaning,model deployment,end-to-end solution]
---
_Creating a model is the rehearsal; deploying it is the live concert — with no room for off-key notes._
-------------------------------------------------------------------------------------------------------


To successfully roll deep learning models out in the real world, we need to consider many things: from model quality in terms of its performance on the test dataset, how different the data in the wild is compared to the model’s training data, and environments aren’t frozen in time — what happens when they drift? — and that, as time goes by, the data feeding our model will inevitably change.

Jeremy Howard, who wrote _Deep Learning for Coders with fastai and PyTorch_, came up with something he calls the **Drivetrain Approach** — a way to make sure our models don’t just work in theory, but deliver real value in practice.

> “Many accurate models are of no use to anyone, and many inaccurate models are highly useful.”, Jeremy wrote.

**So, What Is The Drivetrain Approach?**

Simply put, the Drivetrain Approach starts with your problem’s objective, then ask — what actions you can take to meet that objective? Next, identify the data that supports those actions, and finally, build a model that you can use to get the best outcome in terms of your initial objective.

![Deep Learning for Coders with fastai and PyTorch — The Drivetrain Approach.](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Z_0A1u5tr_m2YrkKxAtxQg.png)

Think about autonomous driving. The objective isn’t “build a car that recognizes stop signs” — that’s just one piece of the puzzle. The real goal is safe and efficient travel. So, we start with the objective (_safety + efficiency_), then list the possible actions the car can take (_brake, accelerate, steer_). Next, we ask what data supports those actions (_cameras, LiDAR, radar, maps_). Finally, we build models that use this data to guide the best action in each situation. That’s the Drivetrain Approach in motion.

At the end of the day, the Drivetrain Approach is what keeps us focused on what truly matters. By grounding everything in objectives, actions, and data, we give our models a real chance to solve meaningful problems — whether it’s safer driving, better healthcare, or smarter businesses.

**Forget Linear, Go Spiral**

Chasing the “perfect” dataset is a trap. There’s lots of students, researchers, and even pros wasted months, even years — searching for it. Meanwhile, the ones who just start are already three iterations ahead, learning and improving. _So, what is_ t_he secret_? Think end-to-end. Don’t get lost fine-tuning models, polishing GUI interfaces, or hand-labeling data. Ship the whole pipeline, see what breaks, and focus on the parts that truly shape the outcome. That’s how real progress happens.

![Photo by digitale.de on Unsplash](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*v8XcqqAz68NiMgBT)

By iterating end-to-end, you’ll quickly gain insight into how much data is truly necessary. For example, you might discover **that 200 labeled samples are enough to achieve the performance your application requires** — something you can’t know until you actually go through all the pipeline end-to-end.

**Always Double-Check Your Data?**

Models are only as good as the data they’re trained on — and the real world data is usually packed with hidden biases. Take this example: imagine you want to build an app to detect healthy skin. You scrape search results for “_healthy skin_” and what do you get? The following figure was the results for that search query.

![Deep Learning for Coders with fastai and PyTorch — Data for a healthy skin detector?](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*G46A8JU6TD3rBYa_UEQajQ.png)

Not diversity, not reality — just endless images of young white women touching their faces. Train your model on that data, and you won’t have a _healthy skin detector_ — guess what you’ll have? — it ends up getting a _“young white woman touching her face” detector_ instead.

That’s why it’s critical to think ahead about the kinds of data your application will encounter in practice, and to make sure your training data truly represents them. Otherwise, your model will reflect not the truth — but the bias baked into its data.

**It’s Very Likely That You Used Wrong Data Augmentation Techniques**

**Data augmentation** is all about giving your training data a makeover — changing its appearance without changing its meaning. For images, that could mean rotating, flipping, tweaking brightness or contrast, or even warping the perspective.

But wait — data augmentation isn’t all sunshine and accuracy boosts. Some techniques can actually _hurt_ your model if used carelessly. Here are some examples:

![Deep Learning for Coders with fastai and PyTorch — Data Augmentation with Crop method.](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*r0fdFVVTAhsptFLrKE1ULg.png)

Data augmentation with cropping, as shown in the figure above, can **chop out crucial details**. Imagine that you’re classifying dog or cat breeds, you might lose the very face or body part that distinguishes one breed from another.

![Deep Learning for Coders with fastai and PyTorch — Data Augmentation with Squish method.](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*K0KmESF8ltPFY3909LiaAA.png)

Next, with squishing or stretching, as shown in the figure above, that **creates shapes that don’t exist in the real world**. As a result, the model would learn warped versions of reality and perform worse.

![Deep Learning for Coders with fastai and PyTorch — Data Augmentation with Pad method.](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*1UYvqQ27jyppZspRudnZnw.png)

Finally, data augmentation with padding, as shown in the figure above, that might seem harmless, but it comes with a hidden cost. All that extra empty space is nothing but wasted computation for your model. Even worse, **it reduces the effective resolution of the part of the image** that really matters — the actual content you want the model to learn from.

**So, what’s the _right_ way to handle this?** Instead of cropping, padding or stretching, the usual practice is to **randomly crop** **different parts of an image** during training. Each time the model sees an image, it gets a slightly different view. Over many passes through the dataset, this teaches the model to pay attention to different features — just like in real life, where the same object might show up in photos framed in slightly different ways.

It’s important to remember that a fresh neural network starts from zero. It doesn’t even know that an object rotated by a tiny angle is still the same thing. By training it on images where objects appear at slightly different sizes or positions, we help it learn the basic idea of what an object is and how it can look under different conditions.

So, how is this smarter approach any different from the _“bad crop”_ we talked about earlier?

*   **Naive cropping**: You slice off parts of the image and hope for the best. But in doing so, you might lose the very clues that matter — a dog’s face, a cat’s ears, the detail that separates one breed from another. The model ends up handicapped because it never gets to see the whole story.
*   **Random, repeated cropping during training**: Instead of permanently cutting things out, you give the model a fresh view each time. One pass, it might see the top of the image. Next pass, the bottom. Over time, it sees the whole object, just framed in different ways. This mirrors real life — objects shift, zoom, and sometimes only show partly in photos. The result? A model that’s tougher, smarter, and far better at recognizing what truly matters.

**Your Model Isn’t Just a Learner, It’s a Cleaner**

Assuming we are training a bear classifier, and now we have already prepared all our data. Time to train the model. _Note: We’ll use the random-cropping augmentation technique mentioned above to train the model._

Now evaluate its performance using something called [confusion matrix](https://medium.com/@tam.tamanna18/understanding-the-confusion-matrix-and-its-applications-3a1c33d799af), that then showed as follow:

![Deep Learning for Coders with fastai and PyTorch — Confusion Matrix for Bear detector.](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*JU9VL6IYmrOv3rYWMCY23Q.png)

This handy chart doesn’t just tell us how many predictions were right or wrong — it shows us _where_ the mistakes are happening. That’s crucial, because errors can come from two very different sources:

*   **Dataset problem** — for example, images that aren’t bears at all, or are labeled incorrectly.
*   **Model problem** — perhaps the model isn’t handling images taken with unusual lighting, or from a different angle, etc.

To dig deeper, we sort the images by their **loss.** We don’t cover how to calculate the loss here, but just knowing that the loss is a number that is higher: if the model is incorrect and confident of its incorrect answer, or if it’s correct but not confident of its correct answer.

![Deep Learning for Coders with fastai and PyTorch — Top images with highest loss.](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*GuOxWF_Ai3hQniHNOSqS0A.png)

As showed in the figure above, the very first high-loss image tells the story. The model swears it’s a _grizzly_, but our dataset insists it’s a _black bear_. One look, and you can tell the model’s probably right. By re-labeling this, we instantly make the dataset cleaner. **While most people assume data cleaning must happen before training**, this example shows that a trained model can actually _point out the messy parts of our dataset_ — faster and more reliably. That’s why a practical strategy is:

1.  Train a quick and simple model first.
2.  Use it to help us with data cleaning.
3.  Retrain our model.

**So, the takeaway is clear:** the model doesn’t just learn from the data — it helps _improve_ _the data_.

**How To Avoid Disaster?**

When you’re building regular software, you can trace every line, inspect every step, and know exactly how the system will behave. Neural networks? Not so much. Their “logic” isn’t hard-coded — it _emerges_ from the patterns in the training data. And that’s where things can get messy.

Imagine rolling out a bear detection system for campsites in national parks. Sounds great, right? But If we used a model trained with a neat little dataset scraped from the internet, there would be all kinds of problems in practice, such as these:

*   Working with video data instead of images
*   Handling nighttime images, which may not appear in this dataset
*   Dealing with low-resolution camera images
*   Ensuring results are returned fast enough to be useful in practice
*   Recognizing bears in positions that are rarely seen in photos that people post online (for example from behind, partially covered by bushes, or a long way away from the camera)

Deep learning are powerful but unpredictable, and that unpredictability can spell disaster. So, how to avoid it? Please keep in mind these following crucial steps to follow when deploying deep learning models:

*   **Collect real data** — don’t rely only on curated internet sets
*   **Start manual and parallel** — let humans double-check the model’s output
*   **Limit scope** — run small, time-boxed trials before scaling
*   **Roll out slowly** — expand step by step, not all at once
*   **Monitor smartly** — track metrics, watch for sudden shifts, and design reports that expose failure modes

**Unforeseen Consequences and Feedback Loops —** Rolling out a model doesn’t just predict behavior — it can change it. Take predictive policing: algorithms flag “high-crime” areas, police flood those neighborhoods, more arrests get recorded, and the cycle reinforces itself. As one study put it: these models aren’t predicting crime, they’re predicting future policing.

**How to Stay Safe —** Before launching any ML system, ask: What if it works really, really well? Who benefits? Who suffers? What’s the worst-case scenario? Then build in careful rollout plans, strong monitoring, and — most importantly — human oversight that actually has power to intervene.

**Final thoughts**

In the end, deploying deep learning models isn’t just about accuracy — it’s about responsibility. From biased data and domain shifts to feedback loops that can spiral out of control, the risks are real. But with careful rollout, constant monitoring, and meaningful human oversight, we can turn these risks into manageable challenges.

If you’d like to dive deeper, these insights come from [_Deep Learning for Coders with fastai and PyTorch_](https://amzn.to/46CvX95) by Jeremy Howard and Sylvain Gugger.

By the way, I wrote my **first post introducing this book** and why it matters for anyone interested in practical AI. You can read it here: [Deep learning for coders with fastai and PyTorch — Build your AI applications without a PhD](https://medium.com/mr-plan-publication/deep-learning-for-coders-with-fastai-and-pytorch-build-your-ai-applications-without-a-phd-3f99b22273e9). For now, we’d love your thoughts and feedback.

📘 Thanks for reading! I’m **Khanh Do Van (k-dovan)**, writing about **AI, NLP, and Data** with a focus on practical insights and real-world applications.
⭐ If you found this useful, **follow me here on Medium** for more deep dives, tutorials, and lessons learned.
💻 You can also check out my projects and code on GitHub: [github.com/k-dovan](https://github.com/k-dovan).