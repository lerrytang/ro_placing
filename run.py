import argparse
from Simulator import Simulator
from Agent import Agent
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main(args):

    if args.pre_train:
        simulator = Simulator(win_h=args.win_h, win_w=args.win_w,
                              frame_skip=args.frame_skip,
                              env_path=args.env_path,
                              img_dir=args.img_dir)
        agent = Agent(simulator, log_dir=args.log_dir)
        agent.pre_train(max_iter=args.max_iter,
                        batch_size=args.batch_size,
                        init_lr=args.learning_rate,
                        data_file=args.data_file,
                        model_file=args.model_dir)
    else:
        simulator = Simulator(win_h=args.win_h, win_w=args.win_w,
                              frame_skip=args.frame_skip,
                              env_path=args.env_path,
                              img_dir=args.img_dir)
        agent = Agent(simulator, log_dir=args.log_dir)
        agent.run_sim(model_file=args.model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--win_w", nargs=1, default=500, type=int,
                        help="width of window")
    parser.add_argument("--win_h", nargs=1, default=500, type=int,
                        help="height of window")

    parser.add_argument("--data_file",
                        help="pre-training database file")
    parser.add_argument("--img_dir", default="./img",
                        help="directory to save sample images")
    parser.add_argument("--model_dir", default="./model",
                        help="directory to save/load pre-trained model files")
    parser.add_argument("--log_dir", default="./log",
                        help="directory to save tensorflow logs")

    parser.add_argument("--pre_train", action="store_true", default=False,
                        help="whether to pre-train an agent")
    parser.add_argument("--max_iter", default=200000, type=int,
                        help="maximum number of iterations for network training")
    parser.add_argument("--frame_skip", default=3, type=int,
                        help="number of frames to skip in simulation")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="training batch size")
    parser.add_argument("--learning_rate", default=0.0005, type=float,
                        help="initial learning rate")

    # mandatory
    parser.add_argument("env_path",
                        help="path to a model XML file")

    args = parser.parse_args()
    main(args)