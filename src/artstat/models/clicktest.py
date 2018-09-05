import click


@click.group()
@click.option("--a", type=str)
@click.option("--b", type=str)
@click.pass_context
def main(ctx, b, a):
    print("a:", a)
    print("b:", b)


@main.command('blip')
@click.pass_context
def blip(ctx):
    print(ctx.obj)  # print("Ahoy from blip:", ctx.obj['ahoy'])


if __name__ == "__main__":
    main(obj={})
